import multiprocessing

from fuzzy_address_matcher.gradio_app import build_app, main

# Keep a top-level `block` for Hugging Face Spaces.
block = build_app()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
