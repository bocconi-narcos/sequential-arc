import random
import numpy as np
import os # For PYTHONHASHSEED

# try:
#     import torch # For PyTorch seeding if used in project
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False

def set_global_seed(seed: int) -> None:
    """
    Sets the seed for global random number generators used in Python, NumPy,
    and optionally PyTorch (if available and used).
    Also sets PYTHONHASHSEED environment variable for reproducible hash-based operations.

    Args:
        seed: The integer seed value.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed)}.")

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PYTHONHASHSEED environment variable
    # Needs to be set before Python interpreter starts for full effect on dict ordering etc.
    # Setting it here might have limited effect if interpreter is already running,
    # but it's good practice for parts of code that might consult it.
    # For full reproducibility of hash-dependent collections, run script with PYTHONHASHSEED=...
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: After Python 3.7, dict iteration order is guaranteed insertion order.
    # Hash seed mainly affects set order and dict order in <3.7, and some hash algorithms.

    # PyTorch (if installed and relevant to the project)
    # if TORCH_AVAILABLE:
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed) # For all GPUs
    #         # For ensuring deterministic behavior on GPU (can impact performance)
    #         # torch.backends.cudnn.deterministic = True
    #         # torch.backends.cudnn.benchmark = False
    #     print(f"Global seed {seed} set for Python random, NumPy, PYTHONHASHSEED, and PyTorch.")
    # else:
    #     print(f"Global seed {seed} set for Python random, NumPy, and PYTHONHASHSEED.")
    print(f"Global seed {seed} set for Python random, NumPy, and PYTHONHASHSEED environment variable.")
    print("Note: For PYTHONHASHSEED to fully affect dicts/sets, it must be set before Python starts.")


def get_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Creates and returns a new NumPy random number generator (Generator instance),
    optionally seeded. This is preferred over using the global np.random state
    for better control and reproducibility in specific components.

    Args:
        seed: An optional integer seed for the RNG. If None, the RNG is initialized
              randomly (or based on system entropy).

    Returns:
        A NumPy Generator instance.
    """
    return np.random.default_rng(seed)


# Example Usage:
if __name__ == "__main__":
    print("--- Testing Global Seeding ---")
    set_global_seed(42)

    print("Python random module (first 5 numbers after seed 42):")
    print([random.randint(0, 100) for _ in range(5)]) # [81, 0, 13, 54, 46] with seed 42

    print("\nNumPy global random (first 5 numbers after seed 42):")
    # np.random.randint uses the global RandomState, not the new Generator API directly,
    # but np.random.seed() affects this global state.
    print(np.random.randint(0, 100, size=5)) # [51 92 14 71 60] with seed 42 (using legacy np.random.randint)

    # Using the new Generator API (recommended)
    rng_from_global_seed_effect = np.random.default_rng() # Will be affected by np.random.seed() if called before first use
                                                          # However, default_rng() is usually independent of global seed once created.
                                                          # np.random.seed() primes the LEGACY np.random state.
                                                          # For true isolation, always pass a seed to default_rng().
    # print("\nNumPy Generator from default_rng() (potentially affected by global np.random.seed):")
    # print(rng_from_global_seed_effect.integers(0, 100, size=5)) # Behavior depends on when global state was first touched.

    print("\n--- Testing Isolated NumPy RNG with get_numpy_rng ---")
    rng1 = get_numpy_rng(seed=123)
    print("RNG1 (seed 123) first 5 numbers:")
    print(rng1.integers(0, 100, size=5)) # [66 92 98 17 83] with seed 123

    rng2 = get_numpy_rng(seed=123) # New generator with same seed
    print("\nRNG2 (seed 123, new instance) first 5 numbers:")
    print(rng2.integers(0, 100, size=5)) # Should be same as RNG1: [66 92 98 17 83]

    rng3 = get_numpy_rng(seed=999) # New generator with different seed
    print("\nRNG3 (seed 999) first 5 numbers:")
    print(rng3.integers(0, 100, size=5)) # Different sequence, e.g. [39 86 46 98 63]

    rng4_unseeded = get_numpy_rng() # Unseeded
    print("\nRNG4 (unseeded) first 5 numbers:")
    print(rng4_unseeded.integers(0, 100, size=5)) # Should be different each run (or from system entropy)

    print(f"\nPYTHONHASHSEED set to: {os.environ.get('PYTHONHASHSEED')}")
    assert os.environ.get('PYTHONHASHSEED') == "42"

    # Verify global numpy state again (should not be affected by isolated generators)
    print("\nNumPy global random again (should continue its own sequence if used, or be re-seeded if np.random.seed called again):")
    # If np.random.seed(42) was the last global call, this continues from that.
    print(np.random.randint(0, 100, size=5)) # Continues from [51 92 14 71 60] -> [20 82 86 74 74]

    print("\nDone with random utilities tests.")
