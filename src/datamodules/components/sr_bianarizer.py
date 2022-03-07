import os
import glob
import random
import pickle
from multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np
import librosa
import traceback

from src.datamodules.components import dataset_utils


def chunked_multiprocess_run(map_func, args, num_workers=None, ordered=True, init_ctx_func=None, q_max_size=1000):
    args = zip(range(len(args)), args)
    args = list(args)
    n_jobs = len(args)
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
    results_queues = []
    if ordered:
        for i in range(num_workers):
            results_queues.append(Queue(maxsize=q_max_size // num_workers))
    else:
        results_queue = Queue(maxsize=q_max_size)
        for i in range(num_workers):
            results_queues.append(results_queue)
    workers = []
    for i in range(num_workers):
        args_worker = args[i::num_workers]
        p = Process(target=chunked_worker, args=(
            i, map_func, args_worker, results_queues[i], init_ctx_func), daemon=True)
        workers.append(p)
        p.start()
    for n_finished in range(n_jobs):
        results_queue = results_queues[n_finished % num_workers]
        job_idx, res = results_queue.get()
        assert job_idx == n_finished or not ordered, (job_idx, n_finished)
        yield res
    for w in workers:
        w.join()
        w.close()


def chunked_worker(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    for job_idx, arg in args:
        try:
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))


class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})


class DapsSRBinarizer:
    def __init__(self, data_dir, binary_data_dir, binarization_args, wav2spec_args, test_num):

        self.data_dir = data_dir
        self.binary_data_dir = binary_data_dir
        self.binarization_args = binarization_args
        self.wav2spec_args = wav2spec_args
        self.test_num = test_num

        self.wavfns = sorted(glob.glob(f'{self.data_dir}/wav48/*/*.flac'))
        self.item2wavfn = {}
        for id, wavfn in enumerate(self.wavfns):
            self.item2wavfn[id] = wavfn
        self.item_names = list(range(len(self.wavfns)))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        return self.item_names[self.test_num:]

    @property
    def valid_item_names(self):
        return self.item_names[:self.test_num]

    @property
    def test_item_names(self):
        return self.valid_item_names #TODO: val=test

    def get_wav_fns(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names

        for item_name in item_names:
            wav_fn = self.item2wavfn[item_name]
            yield item_name, wav_fn

    def process(self):
        os.makedirs(self.binary_data_dir, exist_ok=True)
        target_sr = self.binarization_args['resampled_rate']
        target_dir = f'{self.binary_data_dir}/{str(target_sr)}'
        os.makedirs(target_dir, exist_ok=True)

        self.process_data(target_dir, 'valid')
        self.process_data(target_dir, 'test')
        self.process_data(target_dir, 'train')

    def process_data(self, target_dir, prefix):
        builder = IndexedDatasetBuilder(f'{target_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0

        meta_data = list(self.get_wav_fns(prefix))
        args = [
            list(m) + [self.binarization_args, self.wav2spec_args] for m in meta_data
        ]
        num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3))
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))
        ):
            if item is None:
                continue
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            if not self.binarization_args['with_resample'] and 'resampled_wav' in item:
                del item['resampled_wav']

            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{target_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{target_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f'| {prefix} total duration: {total_sec: .3f}s')

    @classmethod
    def process_item(cls, item_name, wav_fn, binarization_args, wav2spec_args):
        wav, mel = dataset_utils.wav2spec(wav_fn, wav2spec_args)

        res = {
            'item_name': item_name,
            'wav': wav,
            'sec': len(wav) / wav2spec_args['audio_sample_rate'], #TODO check sr
            'len': mel.shape[0],
            'wav_fn': wav_fn,
        }

        if binarization_args['with_resample']:
            sr = wav2spec_args['audio_sample_rate']
            sr_hat = binarization_args['resampled_rate']
            res['resampled_wav'] = librosa.resample(wav, sr, sr_hat)

        return res


class VFSRBinarizer:
    def __init__(self, data_dir, binary_data_dir, binarization_args, wav2spec_args, test_spkers):

        self.data_dir = data_dir
        self.binary_data_dir = binary_data_dir
        self.binarization_args = binarization_args
        self.wav2spec_args = wav2spec_args

        self.wavfns = []

        spkers_train = set(os.listdir(f'{self.data_dir}/wav48')) - set(test_spkers)
        for spker in sorted(spkers_train):
            self.wavfns += sorted(glob.glob(f'{self.data_dir}/wav48/{spker}/*.flac'))

        self.item2wavfn = {id:wavfn for id, wavfn in enumerate(self.wavfns)}
        self.item_names = list(range(len(self.wavfns)))

    @property
    def train_item_names(self):
        return self.item_names

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def get_wav_fns(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names

        for item_name in item_names:
            wav_fn = self.item2wavfn[item_name]
            yield item_name, wav_fn

    def process(self):
        os.makedirs(self.binary_data_dir, exist_ok=True)
        target_sr = self.binarization_args['resampled_rate']
        target_dir = f'{self.binary_data_dir}/{str(target_sr)}'
        os.makedirs(target_dir, exist_ok=True)

        # self.process_data(target_dir, 'valid')
        # self.process_data(target_dir, 'test')
        self.process_data(target_dir, 'train')

    def process_data(self, target_dir, prefix):
        builder = IndexedDatasetBuilder(f'{target_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0

        meta_data = list(self.get_wav_fns(prefix))
        args = [
            list(m) + [self.binarization_args, self.wav2spec_args] for m in meta_data
        ]
        num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3))
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))
        ):
            if item is None:
                continue
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            if not self.binarization_args['with_resample'] and 'resampled_wav' in item:
                del item['resampled_wav']

            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{target_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{target_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f'| {prefix} total duration: {total_sec: .3f}s')

    @classmethod
    def process_item(cls, item_name, wav_fn, binarization_args, wav2spec_args):
        wav, mel = dataset_utils.wav2spec(wav_fn, wav2spec_args)
        res = {
            'item_name': item_name,
            'wav': wav,
            'sec': len(wav) / wav2spec_args['audio_sample_rate'], #TODO check sr
            'len': mel.shape[0],
            'wav_fn': wav_fn,
        }

        if binarization_args['with_resample']:
            sr = wav2spec_args['audio_sample_rate']
            sr_hat = binarization_args['resampled_rate']
            res['resampled_wav'] = librosa.resample(wav, sr, sr_hat)

        return res
