from tools.gradio_app import build_app, main

# Keep a top-level `block` for Hugging Face Spaces.
block = build_app()

if __name__ == "__main__":
    main()
