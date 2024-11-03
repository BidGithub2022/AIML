# AIML

Prerequisites:
On MAC:
Install brew:  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Install python3: /opt/homebrew/bin/brew install python3

1. Mini Conda install: Miniconda (a lightweight version of Anaconda), follow these steps. We'll go over setting up the environment, downloading a model with Hugging Face Transformers, and running it interactively.
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

Initialize conda: conda init

2. Create and Activate a Conda Environment
   conda create -n llm_env python=3.10
   conda activate llm_env

3. Install Required Packages: Install torch with MPS (Metal Performance Shaders) support, which is optimized for Apple Silicon, and the Hugging Face Transformers library:
   pip install torch torchvision transformers

4. Create a Python Script to Download and Run the Model
   vi run_llm.py

   import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Choose the model (GPT-2 in this example)
    model_name = "gpt2"  # You can change this to other models like "gpt2-medium" or "distilgpt2"
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the device to MPS (Metal Performance Shaders) if available for Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    print("Interactive mode. Type 'exit' to quit.\n")

    # Interactive loop
    while True:
        # Prompt the user for input text
        input_text = input("You: ")
        
        # Exit if the user types 'exit'
        if input_text.lower() == "exit":
            print("Exiting the interactive mode.")
            break

        # Prepare inputs and generate response
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
        
        # Decode and print the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Model:", generated_text, "\n")

if __name__ == "__main__":
    main()

5. Run the Python Script
   python3 run_llm.py







