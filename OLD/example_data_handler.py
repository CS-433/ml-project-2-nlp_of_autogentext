from dataset_handler import TrainTestDataset, CrossValDataset

data_dir = "../snips_slu_data_v1.0/smart-lights-en-far-field/"  #"/path/to/smart-lights/data/folder"
dataset = CrossValDataset.from_dir(data_dir)

print(dataset.get_audio_file("Set lights to twenty two percent in the basement"))
print(dataset.get_labels_from_text("Set lights to twenty two percent in the basement"))
    
print(dataset.get_transcript("0.wav"))
print(dataset.get_labels_from_wav("0.wav"))

data_dir = "../snips_slu_data_v1.0/smart-speaker-en-far-field/" #"/path/to/smart-speaker/data/folder"
dataset = TrainTestDataset.from_dir(data_dir)

print(dataset.get_audio_file("I'd like to listen to Drake"))
print(dataset.get_labels_from_text("I'd like to listen to Drake"))

print(dataset.get_transcript("0.wav"))
print(dataset.get_labels_from_wav("0.wav"))
