librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}
aishell1_datasets = {
    "train": ["data_aishell/wav/train"],
    "dev": ["data_aishell/wav/dev"],
    "test": ["data_aishell/wav/test"],
}

magicdata_datasets = {
    "train": ["MagicData/train"],
    "dev": ["MagicData/dev"],
    "test": ["MagicData/test"],
}

aidatatang_datasets = {
    "train": ["aidatatang_200zh/corpus/train"],
    "dev": ["aidatatang_200zh/corpus/dev"],
    "test": ["aidatatang_200zh/corpus/test"],
}

thchs30_datasets = {
    "train": ["data_thchs30/train"],
    "dev": ["data_thchs30/dev"],
    "test": ["data_thchs30/test"],
}

mozilla_datasets = {
    "train": "Mozilla/train.tsv",
    "dev": "Mozilla/test.tsv",
    "test": "Mozilla/train.tsv",
    "validated": "Mozilla/validated.tsv",
}

stcmds_datasets = "ST-CMDS-20170001_1-OS"

primewords_datasets = "primewords_md_2018_set1"

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
