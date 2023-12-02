import os


google_drive_paths = {
    "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
    
    "stylegan2-ffhq-config-f.pkl": "https://drive.google.com/uc?id=1d4x0mQNCb45lWhgITEtGIrCFsptnKyc6",
    "stylegan2-car-config-f.pkl": "https://drive.google.com/uc?id=1PiwJBy348PCwje1x3wj_ZhwGr2UyzkC7",
    "stylegan2-church-config-f.pkl": "https://drive.google.com/uc?id=1JG_Bbm9UV7Gg53PUmD_1MzScfNVRGWVK",
    "afhqdog.pkl": "https://drive.google.com/uc?id=1vTp3M5vjzgSCiq_qxaNz6P8EAZfc3u5o",
    "afhqcat.pkl": "https://drive.google.com/uc?id=1Hfd68lT1HGbb2ZZPc4AyvmKr6jf4W8cC",
    "afhqwild.pkl": "https://drive.google.com/uc?id=1XrDvKlM-5UqdhPQ2-C2F9HGF3C3ViAUO",
    "wikiart-1024-stylegan3-t-17.2Mimg.pkl": "https://drive.google.com/uc?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj",
    "lhq-256-stylegan3-t-25Mimg.pkl": "https://drive.google.com/uc?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1",
    
    "mapper/pretrained/afro.pt": "https://drive.google.com/uc?id=1i5vAqo4z0I-Yon3FNft_YZOq7ClWayQJ",
    "mapper/pretrained/angry.pt": "https://drive.google.com/uc?id=1g82HEH0jFDrcbCtn3M22gesWKfzWV_ma",
    "mapper/pretrained/beyonce.pt": "https://drive.google.com/uc?id=1KJTc-h02LXs4zqCyo7pzCp0iWeO6T9fz",
    "mapper/pretrained/bobcut.pt": "https://drive.google.com/uc?id=1IvyqjZzKS-vNdq_OhwapAcwrxgLAY8UF",
    "mapper/pretrained/bowlcut.pt": "https://drive.google.com/uc?id=1xwdxI2YCewSt05dEHgkpmmzoauPjEnnZ",
    "mapper/pretrained/curly_hair.pt": "https://drive.google.com/uc?id=1xZ7fFB12Ci6rUbUfaHPpo44xUFzpWQ6M",
    "mapper/pretrained/depp.pt": "https://drive.google.com/uc?id=1FPiJkvFPG_y-bFanxLLP91wUKuy-l3IV",
    "mapper/pretrained/hilary_clinton.pt": "https://drive.google.com/uc?id=1X7U2zj2lt0KFifIsTfOOzVZXqYyCWVll",
    "mapper/pretrained/mohawk.pt": "https://drive.google.com/uc?id=1oMMPc8iQZ7dhyWavZ7VNWLwzf9aX4C09",
    "mapper/pretrained/purple_hair.pt": "https://drive.google.com/uc?id=14H0CGXWxePrrKIYmZnDD2Ccs65EEww75",
    "mapper/pretrained/surprised.pt": "https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI",
    "mapper/pretrained/taylor_swift.pt": "https://drive.google.com/uc?id=10jHuHsKKJxuf3N0vgQbX_SMEQgFHDrZa",
    "mapper/pretrained/trump.pt": "https://drive.google.com/uc?id=14v8D0uzy4tOyfBU3ca9T0AzTt3v-dNyh",
    "mapper/pretrained/zuckerberg.pt": "https://drive.google.com/uc?id=1NjDcMUL8G-pO3i_9N6EPpQNXeMc3Ar1r",

    "example_celebs.pt": "https://drive.google.com/uc?id=1VL3lP4avRhz75LxSza6jgDe-pHd2veQG"
}


def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

def llm_setup():
    if not os.path.exists('./llama-2-7b-chat.ggmlv3.q4_1.bin'):
        os.system('wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_1.bin')
    os.system('pip install langchain llama-cpp-python==0.1.78')
    model_path = "./llama-2-7b-chat.ggmlv3.q4_1.bin"

    from langchain.llms import LlamaCpp
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=2000,
        top_p=1)
    return llm

def prompt_refiner(llm, text_prompt, description):
    prompt = f"""
    Context: Modify the text in the following (text, class) tuple to be a relevant description relevant to the class. State your final answer only, without explanation.
    Examples:
        (a happy sky, face) => a happy girl
        (a sad face, face) => a sad face
        (a colorful boy, face) => a cheerful boy
    Question: ({text_prompt}, {description}) =>
    """
    print(f"Prompt to LLM: {prompt}")
    text_prompt = llm(prompt).split(':')[-1].strip()
    return text_prompt

llm = llm_setup()