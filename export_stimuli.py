"""
@Author: Iordanis Thoidis
@Date: 11/5/23
@Link: https://github.com/ithoidis/Perceptual-Speaker-Embeddings-Listening-Test
"""
import os
import numpy as np
import pandas as pd
import scipy.spatial.distance
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from tqdm import tqdm
import random
from utils import read_audio, write_audio, rms, apply_raised_cosine_ramp, silence, attenuate_sound, hook
import os
import tarfile
import shutil
from tqdm import tqdm
import urllib.request


def export_paired_metadata(noise=None, snr=None):
    assert noise in [None, 'babble', 'iltass']
    random.seed(0)
    np.random.seed(0)

    embedding_filepath = 'Audio/speaker_embeddings/embedding_%s_%s.npy' % (noise if noise is not None else 'clean',
                                                                     str(snr) if snr is not None else '')

    # load embeddings dataframe
    df = pd.DataFrame(np.load(embedding_filepath, allow_pickle=True),
                      columns=['fs', 'utterance', 'speaker_id', 'chapter_id', 'utterance_id',
                               'embedding', 'gender', 'F0', 'Spectral Rolloff', 'Spectral Bandwidth',
                               'Spectral Centroid', 'Zero Crossing Rate'])

    speakers = df['speaker_id'].unique().tolist()
    df = df[df['speaker_id'].isin(speakers)]
    speaker_similarities = []
    for speaker in tqdm(speakers, desc='Computing similarities for each speakers'):
        sub_df = df[df['speaker_id'] == speaker].reset_index()
        embeddings = np.array([sub_df['embedding'][i] for i in range(len(sub_df))])
        intra_similarities = np.ones((len(embeddings), len(embeddings)))
        for i_emb in range(len(embeddings)):
            for j_emb in range(len(embeddings)):
                if i_emb != j_emb:
                    intra_similarities[i_emb, j_emb] = 1 - scipy.spatial.distance.cosine(embeddings[i_emb],
                                                                                         embeddings[j_emb])
        thres = []
        for speaker2 in speakers:
            if speaker != speaker2:
                sub_df2 = df[df['speaker_id'] == speaker2].reset_index()
                embeddings_sp1 = np.array([sub_df['embedding'][i] for i in range(len(sub_df))])
                embeddings_sp2 = np.array([sub_df2['embedding'][i] for i in range(len(sub_df2))])
                inter_similarities = np.ones((len(embeddings_sp1), len(embeddings_sp2)))
                for i_emb in range(len(embeddings_sp1)):
                    for j_emb in range(len(embeddings_sp2)):
                        inter_similarities[i_emb, j_emb] = 1 - scipy.spatial.distance.cosine(embeddings_sp1[i_emb],
                                                                                             embeddings_sp2[j_emb])
                # compute auc and find optimal threshold.
                sim = np.concatenate([intra_similarities.ravel(), inter_similarities.ravel()])
                targets = [1] * len(intra_similarities.ravel()) + [0] * len(inter_similarities.ravel())
                precision, recall, _ = precision_recall_curve(targets, sim)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(targets, sim)
                fpr, tpr, thresholds = roc_curve(targets, sim)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                assert not np.isnan(optimal_threshold)
                thres.append(optimal_threshold)
                accuracy = np.mean(1 * targets == 1 * (sim >= optimal_threshold))
                speaker_similarities.append({'speaker1': speaker, 'speaker2': speaker2, 'pr_auc': pr_auc,
                                             'roc_auc': roc_auc, 'accuracy': accuracy, 'threshold': optimal_threshold})
    pd.DataFrame(speaker_similarities).to_pickle('Audio/speaker_embeddings/speaker_similarities_%s_%s.pkl' %
                                                 (noise if noise is not None else 'clean', str(snr)
                                                  if snr is not None else ''))

    # -------------------- export metadata file -----------------------
    # select the N less similar and the M most similar samples from each speaker
    N, M = 1, 1
    # select the K less similar and the L most similar samples between the two speakers
    K, L = 1, 3

    # 5 categories in [0, 1] classification accuracy
    embedding_filepath = 'Audio/speaker_embeddings/embedding_%s_%s.npy' % (
                        noise if noise is not None else 'clean', str(snr) if snr is not None else '')
    # load embeddings dataframe
    df = pd.DataFrame(np.load(embedding_filepath, allow_pickle=True),
                      columns=['fs', 'utterance', 'speaker_id', 'chapter_id', 'utterance_id',
                               'embedding', 'gender', 'F0', 'Spectral Rolloff', 'Spectral Bandwidth',
                               'Spectral Centroid', 'Zero Crossing Rate'])
    speaker_similarities_all = pd.read_pickle('Audio/speaker_embeddings/speaker_similarities_%s_%s.pkl' %
                                              (noise if noise is not None else 'clean',
                                               str(snr) if snr is not None else ''))
    # remove duplicates
    speaker_similarities = speaker_similarities_all.sort_values('pr_auc', ascending=True)
    speaker_similarities = speaker_similarities.drop_duplicates('speaker1').drop_duplicates('speaker2').reset_index()
    rows_to_drop = []
    for i_sim in range(len(speaker_similarities)):
        for j_sim in range(i_sim, len(speaker_similarities)):
            if speaker_similarities['speaker1'][i_sim] == speaker_similarities['speaker2'][j_sim]:
                rows_to_drop.append(j_sim)

    speaker_similarities = speaker_similarities.drop(rows_to_drop).reset_index()

    # remove duplicate speakers across columns
    number_of_pairs = 10
    used_filenames = []
    sample_pairs = []
    for pair_id in tqdm(range(number_of_pairs)):
        speaker1, speaker2 = speaker_similarities['speaker1'][pair_id], speaker_similarities['speaker2'][pair_id],
        # get the N most similar and the M least similar samples between the two speakers
        sub_df1 = df[df['speaker_id'] == speaker1].reset_index()
        sub_df2 = df[df['speaker_id'] == speaker2].reset_index()
        embeddings_sp1 = np.array([sub_df1['embedding'][i] for i in range(len(sub_df1))])
        embeddings_sp2 = np.array([sub_df2['embedding'][i] for i in range(len(sub_df2))])
        filenames1 = ['%d-%d-%.4d.flac' % (sub_df1['speaker_id'][i], sub_df1['chapter_id'][i],
                                          sub_df1['utterance_id'][i])
                      for i in range(len(sub_df1))]
        filenames2 = ['%d-%d-%.4d.flac' % (sub_df2['speaker_id'][i], sub_df2['chapter_id'][i],
                                          sub_df2['utterance_id'][i])
                      for i in range(len(sub_df2))]

        inter_similarities = np.ones((len(embeddings_sp1), len(embeddings_sp2)))
        for isp1 in range(len(embeddings_sp1)):
            for jsp2 in range(len(embeddings_sp2)):
                inter_similarities[isp1, jsp2] = 1 - scipy.spatial.distance.cosine(embeddings_sp1[isp1],
                                                                                   embeddings_sp2[jsp2])

        # select the K samples with the lowest similarity
        temp_sim = inter_similarities.copy()
        np.fill_diagonal(temp_sim, 1.)
        nt = 0
        last_idx1, last_idx2 = -1, -1
        while nt < K:
            idx1, idx2 = np.unravel_index(temp_sim.argmin(), temp_sim.shape)
            if last_idx1 == idx1 and last_idx2 == idx2:
                nt = K
                print('list emptied!1')
                continue
            last_idx1, last_idx2 = idx1, idx2
            if filenames1[idx1] not in used_filenames and filenames2[idx2] not in used_filenames:
                used_filenames.append(filenames1[idx1])
                used_filenames.append(filenames2[idx2])
                # how many times does speaker1 and speaker2 are in the same order
                idx1_order1 = np.sum([sample_pairs[i]['speech_id1'].split('-')[0] == filenames1[idx1].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx1_order2 = np.sum([sample_pairs[i]['speech_id2'].split('-')[0] == filenames1[idx1].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx2_order1 = np.sum([sample_pairs[i]['speech_id1'].split('-')[0] == filenames2[idx2].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx2_order2 = np.sum([sample_pairs[i]['speech_id2'].split('-')[0] == filenames2[idx2].split('-')[0]
                                      for i in range(len(sample_pairs))])
                if (idx1_order1 - idx1_order2) + (idx2_order2 - idx2_order1) > 0:
                    f1, f2 = filenames2[idx2], filenames1[idx1]
                else:
                    f1, f2 = filenames1[idx1], filenames2[idx2]
                prediction = 1 * (temp_sim[idx1, idx2] >= speaker_similarities['threshold'][pair_id])

                sample_pairs.append({'speech_id1': f1, 'speech_id2': f2,
                                     'similarity': temp_sim[idx1, idx2], 'prediction': prediction})
                nt = nt + 1
            temp_sim[idx1, idx2] = 1.

        # select the L samples with the highest similarity
        temp_sim = inter_similarities.copy()
        np.fill_diagonal(temp_sim, 0.)
        nt = 0
        last_idx1, last_idx2 = -1, -1
        while nt < L:
            idx1, idx2 = np.unravel_index(temp_sim.argmax(), temp_sim.shape)
            if (last_idx1 == idx1) and (last_idx2 == idx2):
                nt = L
                print('list emptied!2')
                continue
            if (not (filenames1[idx1] in used_filenames)) and (not (filenames2[idx2] in used_filenames)):
                used_filenames.append(filenames1[idx1])
                used_filenames.append(filenames2[idx2])
                # how many times does speaker1 and speaker2 are in the same order
                idx1_order1 = np.sum([sample_pairs[i]['speech_id1'].split('-')[0] == filenames1[idx1].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx1_order2 = np.sum([sample_pairs[i]['speech_id2'].split('-')[0] == filenames1[idx1].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx2_order1 = np.sum([sample_pairs[i]['speech_id1'].split('-')[0] == filenames2[idx2].split('-')[0]
                                      for i in range(len(sample_pairs))])
                idx2_order2 = np.sum([sample_pairs[i]['speech_id2'].split('-')[0] == filenames2[idx2].split('-')[0]
                                      for i in range(len(sample_pairs))])
                if (idx1_order1 - idx1_order2) + (idx2_order2 - idx2_order1) > 0:
                    f1, f2 = filenames2[idx2], filenames1[idx1]
                else:
                    f1, f2 = filenames1[idx1], filenames2[idx2]
                prediction = 1 * (temp_sim[idx1, idx2] >= speaker_similarities['threshold'][pair_id])
                sample_pairs.append({'speech_id1': f1, 'speech_id2': f2,
                                     'similarity': temp_sim[idx1, idx2], 'prediction': prediction})
                nt = nt + 1
            last_idx1, last_idx2 = idx1.copy(), idx2.copy()
            temp_sim[idx1, idx2] = 0.

        # for each speaker individually
        # get the K most similar and the L least similar samples
        for speaker in [speaker1, speaker2]:
            sub_df = df[df['speaker_id'] == speaker].reset_index()
            filenames = ['%d-%d-%.4d.flac' % (sub_df['speaker_id'][i], sub_df['chapter_id'][i],
                                             sub_df['utterance_id'][i]) for i in range(len(sub_df))]
            embeddings = np.array([sub_df['embedding'][i] for i in range(len(sub_df))])
            intra_similarities = np.ones((len(embeddings), len(embeddings)))
            threshold = speaker_similarities_all[speaker_similarities_all['speaker1'] == speaker]['threshold'].median()
            assert not np.isnan(threshold)
            for i_emb in range(len(embeddings)):
                for j_emb in range(len(embeddings)):
                    if i_emb != j_emb:
                        intra_similarities[i_emb, j_emb] = 1 - scipy.spatial.distance.cosine(embeddings[i_emb],
                                                                                             embeddings[j_emb])

            # select the N samples from the same speaker with the lowest similarity
            temp_sim = intra_similarities.copy()
            np.fill_diagonal(temp_sim, 1.)
            nt = 0
            last_idx1, last_idx2 = -1, -1
            while nt < N:
                idx1, idx2 = np.unravel_index(temp_sim.argmin(), temp_sim.shape)
                if last_idx1 == idx1 and last_idx2 == idx2:
                    nt = N
                    continue
                last_idx1, last_idx2 = idx1, idx2
                if filenames[idx1] not in used_filenames and filenames[idx2] not in used_filenames:
                    used_filenames.append(filenames[idx1])
                    used_filenames.append(filenames[idx2])
                    prediction = 1 * (temp_sim[idx1, idx2] >= threshold)
                    sample_pairs.append({'speech_id1': filenames[idx1], 'speech_id2': filenames[idx2],
                                         'similarity': temp_sim[idx1, idx2], 'prediction': prediction})
                    nt = nt + 1
                temp_sim[idx1, idx2], temp_sim[idx2, idx1] = 1., 1.

            # select the M samples from the same speaker with the highest similarity
            temp_sim = intra_similarities.copy()
            np.fill_diagonal(temp_sim, 0.)
            nt = 0
            last_idx1, last_idx2 = -1, -1
            while nt < M:
                idx1, idx2 = np.unravel_index(temp_sim.argmax(), temp_sim.shape)
                if last_idx1 == idx1 and last_idx2 == idx2:
                    nt = M
                    continue
                last_idx1, last_idx2 = idx1, idx2
                if filenames[idx1] not in used_filenames and filenames[idx2] not in used_filenames:
                    used_filenames.append(filenames[idx1])
                    used_filenames.append(filenames[idx2])
                    prediction = 1 * (temp_sim[idx1, idx2] >= threshold)
                    sample_pairs.append({'speech_id1': filenames[idx1], 'speech_id2': filenames[idx2],
                                         'similarity': temp_sim[idx1, idx2], 'prediction': prediction})
                    nt = nt + 1
                temp_sim[idx1, idx2], temp_sim[idx2, idx1] = 0., 0.

    # randomize the order of the trials
    random.shuffle(sample_pairs)

    # make sure each sample should exist only once in the final list
    sample_pairs = pd.DataFrame(sample_pairs)
    same_speaker = np.sum([sample_pairs['speech_id1'][i].split('-')[0] == sample_pairs['speech_id2'][i].split('-')[0]
                           for i in range(len(sample_pairs))])
    diff_speaker = np.sum([sample_pairs['speech_id1'][i].split('-')[0] != sample_pairs['speech_id2'][i].split('-')[0]
                           for i in range(len(sample_pairs))])

    # normalize order effect
    all_speakers = np.unique(np.ravel([[sample_pairs['speech_id1'][i].split('-')[0],
                                        sample_pairs['speech_id2'][i].split('-')[0]]
                                       for i in range(len(sample_pairs))]))
    print('Checking presentation order balance for each speaker (first-second)')
    for speaker in all_speakers:
        # measure how many times is in the order 1 and 2.
        idx1_order1 = np.sum([sample_pairs['speech_id1'][i].split('-')[0] == speaker
                              for i in range(len(sample_pairs))])
        idx1_order2 = np.sum([sample_pairs['speech_id2'][i].split('-')[0] == speaker
                              for i in range(len(sample_pairs))])
        print('Speaker:', speaker, 'Order %d-%d' % (int(idx1_order1), int(idx1_order2)))
    print('Total samples:', len(sample_pairs))
    print('Same speaker: %d - Different speaker: %d' % (int(same_speaker), int(diff_speaker)))
    sample_pairs.to_pickle('Audio/speaker_embeddings/speaker_sample_pairs_%s_%s.pkl' %
                           (noise if noise is not None else 'clean', str(snr) if snr is not None else ''))


def export_audio_files(noise=None, snr=None, reverse_order=False):
    root = 'Audio/LibriSpeech'
    filepath = 'Audio/speaker_embeddings/speaker_sample_pairs_%s_%s.pkl' % (noise if noise is not None else 'clean',
                                                                            str(snr) if snr is not None else '')
    meta = pd.read_pickle(filepath).to_dict('records')
    ear = 'both'
    fs = 16000
    level = 0.
    segment = 3.
    interval = 1.5
    trials = 40
    sessions = 2
    x_n = []

    if noise is not None:
        assert snr is not None
        _, x_n = read_audio('Audio/Noises/%s.wav' % noise, target_fs=16000)

    def stimulus(x1, x2, interval_duration, stim_fs=16000, stim_level=-10.0, stim_ear='both', fs=16000):
        x1 = x1 / rms(x1) * 0.05
        x2 = x2 / rms(x2) * 0.05

        x1 = apply_raised_cosine_ramp(x1, stim_fs, onset_duration_ms=50, offset_duration_ms=50)
        x2 = apply_raised_cosine_ramp(x2, stim_fs, onset_duration_ms=50, offset_duration_ms=50)
        if np.random.randint(0, 1):
            x_prim = np.concatenate((x1, silence(interval_duration, stim_fs), x2), axis=0)
        else:
            x_prim = np.concatenate((x2, silence(interval_duration, stim_fs), x1), axis=0)

        if stim_ear == 'left':
            x_left, x_right = x_prim, np.zeros_like(x_prim)
        elif stim_ear == 'right':
            x_left, x_right = np.zeros_like(x_prim), x_prim
        else:
            x_left, x_right = x_prim, x_prim

        x_left = apply_raised_cosine_ramp(x_left, stim_fs, onset_duration_ms=20, offset_duration_ms=20)
        x_right = apply_raised_cosine_ramp(x_right, stim_fs, onset_duration_ms=20, offset_duration_ms=20)
        x_left = attenuate_sound(x_left, stim_level)
        x_right = attenuate_sound(x_right, stim_level)
        x_prim = [x_left, x_right]
        if np.amax(np.abs(x_prim)) >= 1:
            print(np.amax(np.abs(x_left)), np.amax(np.abs(x_right)))
            print('this will clip!')
        return np.array(x_prim).transpose()

    for session in range(sessions):
        for trial in tqdm(range(trials), desc='Session %d' % session):

            _, speech1 = read_audio(os.path.join(root, meta[trial]['speech_id1'].replace('.wav', '.flac')), target_fs=fs)
            _, speech2 = read_audio(os.path.join(root, meta[trial]['speech_id2'].replace('.wav', '.flac')), target_fs=fs)
            speech1 = speech1[:int(fs * segment)]
            speech2 = speech2[:int(fs * segment)]

            if reverse_order:
                speech1, speech2 = speech2, speech1

            if noise is not None:
                # get random noise segment
                noise_start = np.random.randint(0, len(x_n) - segment * fs - 1)
                x_n_segment = x_n[noise_start:int(noise_start+segment*fs)]

                # apply the same noise in both segments in the same SNR.
                pre_snr = np.sqrt(np.mean(speech1 ** 2)) / (np.sqrt(np.mean(x_n_segment ** 2)) + 1e-6)
                scale_factor = 10. ** (-1 * snr / 20.) * pre_snr

                speech1 = speech1[:len(x_n)] + x_n_segment * scale_factor if len(speech1) > len(x_n_segment) \
                    else speech1 + x_n[:len(speech1)] * scale_factor

                pre_snr = np.sqrt(np.mean(speech2 ** 2)) / (np.sqrt(np.mean(x_n_segment ** 2)) + 1e-6)
                scale_factor = 10. ** (-1 * snr / 20.) * pre_snr
                speech2 = speech2[:len(x_n)] + x_n_segment * scale_factor if len(speech2) > len(x_n_segment) \
                    else speech2 + x_n[:len(speech2)] * scale_factor

            x = stimulus(speech1, speech2, interval, stim_fs=fs, stim_level=level, stim_ear=ear)
            x = x[:, 0]
            if not os.path.exists('Audio/stimuli'):
                os.mkdir('Audio/stimuli')
            write_audio('Audio/stimuli/s%d_t%.2d.wav' % (session, trial), fs, x)


def check_data(source_folder='Audio/LibriSpeech'):
    assert os.path.exists('Audio')
    if not os.path.exists(source_folder):
        if not os.path.exists('Audio/test-clean.tar.gz'):
            print('Downloading LibriSpeech test-clean subset (368MB)...')
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc='LibriSpeech') as t:
                urllib.request.urlretrieve('https://www.openslr.org/resources/12/test-clean.tar.gz',
                                           'Audio/test-clean.tar.gz', reporthook=hook(t))

        print('Extracting LibriSpeech audio files...')
        tar = tarfile.open('Audio/test-clean.tar.gz', "r:gz")
        tar.extractall(path='Audio/')
        tar.close()
        os.remove('Audio/test-clean.tar.gz')
        # Move the files to the source directory
        for dirpath, dirnames, filenames in os.walk(source_folder):
            for filename in filenames:
                if '.flac' in filename:
                    filepath = os.path.join(dirpath, filename)
                    new_filename = os.path.join(source_folder, filename)
                    os.rename(filepath, new_filename)
        shutil.rmtree('Audio/LibriSpeech/test-clean')

    if not os.path.exists('Audio/speaker_embeddings/training_clean.pkl') or \
            not os.path.exists('Audio/speaker_embeddings/speaker_sample_pairs_clean_.pkl'):
        export_paired_metadata()

    if not os.path.exists('Results'):
        os.mkdir('Results')

    if not os.path.exists('Results/speaker'):
        os.mkdir('Results/speaker')

    print('Check Done!')


if __name__ == '__main__':
    export_paired_metadata(noise=None)
    export_paired_metadata(noise='iltass', snr=5)