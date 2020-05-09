from multiprocess.pool import ThreadPool
from encoder.params_data import *
from encoder.config import librispeech_datasets, aishell1_datasets, magicdata_datasets, aidatatang_datasets, thchs30_datasets, mozilla_datasets, primewords_datasets, stcmds_datasets, anglophone_nationalites
from datetime import datetime
from encoder import audio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os

class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()
        
        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()
        
    def _log_params(self):
        from encoder import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")
    
    def write_line(self, line):
        self.text_file.write("%s\n" % line)
        
    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)
            
    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()
       
        
def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path, DatasetLog):
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension,
                             skip_existing, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))
    
    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
        
        # Create an output directory with that name, as well as a txt file containing a 
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")
        
        # There's a possibility that the preprocessing was interrupted earlier, check if 
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}
        
        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue
                
            # Load and preprocess the waveform
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue
            
            # Create the mel spectrogram, discard those that are too short
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue
            
            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
        
        sources_file.close()
    
    # Process the utterances for each speaker
    with ThreadPool(8) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)

def _preprocess_speaker_with_files(speaker_dir, dataset_name, datasets_root, out_dir, files_by_speakers,
                             skip_existing, logger):

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(args):
        speaker, files = args
        # Give a name to the speaker that includes its dataset
        parts = list(speaker_dir.relative_to(datasets_root).parts)
        parts.append(speaker)
        speaker_name = "_".join(parts)
        
        # Create an output directory with that name, as well as a txt file containing a 
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")
        
        # There's a possibility that the preprocessing was interrupted earlier, check if 
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}
        
        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath, out_fname in files:
            # Check if the target output file already exists
            if skip_existing and out_fname in existing_fnames:
                continue
                
            # Load and preprocess the waveform
            try:
                wav = audio.preprocess_wav(in_fpath)
                if len(wav) == 0:
                    continue
            except:
                print('wave preprocess error')
                continue
            
            # Create the mel spectrogram, discard those that are too short
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue
            
            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
        
        sources_file.close()
    
    # Process the utterances for each speaker
    items = files_by_speakers.items()
    with ThreadPool(8) as pool:
        list(tqdm(pool.imap(preprocess_speaker, files_by_speakers.items()), dataset_name, len(items),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)

def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in librispeech_datasets["train"]["other"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                                 skip_existing, logger)


def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False):
    # Initialize the preprocessing
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the contents of the meta file
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]
    
    # Select the ID and the nationality, filter out non-anglophone speakers
    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                        nationality.lower() in anglophone_nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
          (len(keep_speaker_ids), len(nationalities)))
    
    # Get the speaker directories for anglophone speakers only
    speaker_dirs = dataset_root.joinpath("wav").glob("*")
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                    speaker_dir.name in keep_speaker_ids]
    print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." % 
          (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))

    # Preprocess all speakers
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger)


def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False):
    # Initialize the preprocessing
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return
    
    # Get the speaker directories
    # Preprocess all speakers
    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "m4a",
                             skip_existing, logger)

def preprocess_aishell1(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in aishell1_datasets["train"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                 skip_existing, logger)

def preprocess_magicdata(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in magicdata_datasets["train"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                 skip_existing, logger)

def preprocess_aidatatang(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in aidatatang_datasets["train"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                 skip_existing, logger)

def preprocess_thchs30(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in thchs30_datasets["train"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return
        
        extension = "wav"
        speaker_dir = datasets_root.joinpath(dataset_name)

        files_by_speakers = {}
        for in_fpath in speaker_dir.glob("*.%s" % extension):
            in_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = in_fname.replace(".%s" % extension, ".npy")
            speaker, serial = out_fname.split('.')[0].split('_')
            if speaker not in files_by_speakers:
                files_by_speakers[speaker] = []
            files_by_speakers[speaker].append([in_fpath, out_fname])
        
        # Preprocess all speakers
        _preprocess_speaker_with_files(speaker_dir, dataset_name, datasets_root, out_dir, files_by_speakers,
                                 skip_existing, logger)

def preprocess_mozilla(datasets_root: Path, out_dir: Path, skip_existing=False):
    ds = mozilla_datasets["validated"]
    ds_file = datasets_root.joinpath(ds)
    dataset_name = os.path.dirname(ds)

    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    extension = "mp3"
    speaker_dir = datasets_root.joinpath(dataset_name).joinpath('clips')

    data = pd.read_csv(ds_file, sep='\t', header=0)

    files_by_speakers = {}
    for index, row in data.iterrows():
        speaker = row['client_id']
        in_fname = row['path']
        in_fpath = speaker_dir.joinpath(in_fname)
        out_fname = in_fname.replace(".%s" % extension, ".npy")
        if speaker not in files_by_speakers:
            files_by_speakers[speaker] = []
        files_by_speakers[speaker].append([in_fpath, out_fname])

    # Preprocess all speakers
    _preprocess_speaker_with_files(speaker_dir, dataset_name, datasets_root, out_dir, files_by_speakers,
                                skip_existing, logger)

def preprocess_primewords(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = primewords_datasets
    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    extension = "wav"
    speaker_dir = datasets_root.joinpath(dataset_name).joinpath('audio_files')

    ds_file = str(datasets_root.joinpath(dataset_name).joinpath('set1_transcript.json'))

    with open(ds_file) as f:
        data=json.load(f)

        files_by_speakers = {}
        for row in data:
            speaker = row['user_id']
            in_fname = row['file']
            in_fpath = speaker_dir.joinpath(in_fname[0]).joinpath(in_fname[0:2]).joinpath(in_fname)
            out_fname = in_fname.replace(".%s" % extension, ".npy")
            if speaker not in files_by_speakers:
                files_by_speakers[speaker] = []
            files_by_speakers[speaker].append([in_fpath, out_fname])

        # Preprocess all speakers
        _preprocess_speaker_with_files(speaker_dir, dataset_name, datasets_root, out_dir, files_by_speakers,
                                 skip_existing, logger)

def preprocess_stcmds(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = stcmds_datasets
    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    extension = "wav"
    speaker_dir = datasets_root.joinpath(dataset_name)

    files_by_speakers = {}
    for in_fpath in speaker_dir.glob("*.%s" % extension):
        in_fname = os.path.basename(in_fpath)
        speaker = in_fname[9:15]
        out_fname = in_fname.replace(".%s" % extension, ".npy")
        if speaker not in files_by_speakers:
            files_by_speakers[speaker] = []
        files_by_speakers[speaker].append([in_fpath, out_fname])

    # Preprocess all speakers
    _preprocess_speaker_with_files(speaker_dir, dataset_name, datasets_root, out_dir, files_by_speakers,
                                skip_existing, logger)