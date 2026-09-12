/*
 * Headless shim over the UNMODIFIED games/2048.c game logic.
 *
 * Why this exists:
 *   - 2048.c's addRandom() seeds libc rand() with time(NULL) on first call via
 *     a function-local `static`, so spawn randomness cannot be seeded,
 *     reproduced, or made thread-safe from outside. That breaks parallel,
 *     reproducible GA rollouts.
 *   - Everything else (slideArray, rotateBoard, moveUp/Left/Down/Right,
 *     gameEnded, countEmpty, findPairDown) is pure, global-state-free logic
 *     and is reused verbatim via #include.
 *
 * Spawn policy replicates addRandom() exactly, except the RNG stream comes
 * from a caller-seeded xorshift32 state (one uint32 per game/episode):
 *   - empty cells are listed scanning x (column) outer, y (row) inner
 *   - first draw picks the position:  draw % len
 *   - second draw picks the value:    (draw % 10) / 9 + 1  -> 2 w.p. 0.9, 4 w.p. 0.1
 *
 * Board memory layout is identical to 2048.c: board[x][y] with x = column,
 * y = row (transposed vs. the usual row-major convention).
 *
 * Direction encoding used by the shim API:
 *   0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
 */
#include "../games/2048.c"

#define SHIM_DIR_UP 0
#define SHIM_DIR_RIGHT 1
#define SHIM_DIR_DOWN 2
#define SHIM_DIR_LEFT 3

#define SHIM_SEED_ZERO 0x9E3779B9u /* xorshift32 is stuck at 0; remap seed 0 */

/* xorshift32 (13, 17, 5). The torch-side implementation in ai2048/vec_env.py
 * must produce the exact same sequence: this is the contract that lets the C,
 * CPU-vectorized and GPU environments be compared bit-for-bit. */
static uint32_t shim_xs32(uint32_t *s)
{
	uint32_t x = *s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*s = x;
	return x;
}

static bool shim_move(uint8_t board[SIZE][SIZE], uint32_t *score, int dir)
{
	switch (dir) {
	case SHIM_DIR_UP: return moveUp(board, score);
	case SHIM_DIR_RIGHT: return moveRight(board, score);
	case SHIM_DIR_DOWN: return moveDown(board, score);
	case SHIM_DIR_LEFT: return moveLeft(board, score);
	default: return false;
	}
}

static void shim_add_random(uint8_t board[SIZE][SIZE], uint32_t *rng)
{
	uint8_t x, y, r, n, len = 0;
	uint8_t list[SIZE * SIZE][2];

	for (x = 0; x < SIZE; x++) {
		for (y = 0; y < SIZE; y++) {
			if (board[x][y] == 0) {
				list[len][0] = x;
				list[len][1] = y;
				len++;
			}
		}
	}
	if (len > 0) {
		r = (uint8_t)(shim_xs32(rng) % len);
		n = (uint8_t)((shim_xs32(rng) % 10) / 9 + 1);
		board[list[r][0]][list[r][1]] = n;
	}
}

/* Zero the board and place the two starting tiles.
 * `rng` is the persistent xorshift32 state: seed 0 is remapped here (the same
 * constant the torch environment uses) and the advanced state is stored back,
 * so the caller keeps a coherent stream across new_game/step. */
void shim_new_game(uint8_t board[SIZE][SIZE], uint32_t *rng)
{
	uint8_t x, y;

	if (*rng == 0) {
		*rng = SHIM_SEED_ZERO;
	}
	for (x = 0; x < SIZE; x++) {
		for (y = 0; y < SIZE; y++) {
			board[x][y] = 0;
		}
	}
	shim_add_random(board, rng);
	shim_add_random(board, rng);
}

/* One headless step, mirroring main()'s sequence: move -> spawn -> ended.
 * Returns a bitmask: bit0 = board changed (move legal), bit1 = game ended
 * after the spawn. When the move is illegal nothing changes and no RNG draws
 * happen, exactly like the original game loop. rng advances in place. */
int shim_step(uint8_t board[SIZE][SIZE], uint32_t *score, int dir, uint32_t *rng)
{
	int moved;

	if (!shim_move(board, score, dir)) {
		return 0;
	}
	moved = 1;
	shim_add_random(board, rng);
	if (gameEnded(board)) {
		moved |= 2;
	}
	return moved;
}

/* Apply a move WITHOUT spawning (for policy lookaheads). `out` receives the
 * resulting board and must not alias `in`. Returns 1 if the move changed the
 * board, 0 otherwise (out is then a copy of in). */
int shim_moved_board(const uint8_t in[SIZE][SIZE], uint8_t out[SIZE][SIZE], int dir)
{
	uint32_t score = 0;

	memcpy(out, in, SIZE * SIZE);
	return shim_move(out, &score, dir) ? 1 : 0;
}

uint8_t shim_game_ended(const uint8_t board[SIZE][SIZE])
{
	/* gameEnded() rotates its argument; work on a copy so the input stays
	 * const. */
	uint8_t copy[SIZE][SIZE];

	memcpy(copy, board, SIZE * SIZE);
	return gameEnded(copy) ? 1 : 0;
}
