from datasets import load_dataset

ds = load_dataset("AI-Lab-Makerere/beans")

ds.save_to_disk("./beans_dataset")