import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm
from os.path import exists
import math
from scipy import stats, signal
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# (import) helper functions
from src.utls import (
    find_handlandmarks,
    mp_hand_labels,
    butter_bandpass_filter,
    pcs2spec,
)

from src.config import (cfg_bandpass_fmin, cfg_bandpass_fmax,
                        cfg_frequency_win_oi_min, cfg_frequency_win_oi_max)

# Initializations: static code
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands


class MpHandAnalyst:
    def __init__(self):
        # when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        # as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        # specified value then again it switches back to detection
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1
        )

        self.cfg_mp_labels = mp_hand_labels()
        self.cfg_mp_max_n_hands = 2
        # array to store ROI if video has been processed, size (2 [min,max] x 2
        # [x,y] x 2[hands])
        self.cfg_mp_roi = np.zeros((2, 2, self.cfg_mp_max_n_hands))
        self.cfg_mp_roi[:] = np.nan
        self.cfg_mp_n_hands = np.nan

    def process_video(self, path_to_video):
        """
        This functions finds the coordinates of (multiple) hand landmarks in a video

        Parameters
        ----------
        path_to_video : string
            Path to video from os module.

        Returns
        -------
        self.mp_accuracy : float
            MP accuaracy score per hand per frame, NaN if not scored

        self.mp_positions_norm : float
            MP normalised position to videos size per landmark per hand, NaN if not determined. Size is n [n_frames] x 63 [landmarks] x 2 [hands]

        self.mp_hand_labels : string
            MP assigned hand label per frame, empty if not applicable

        """

        # For video input:
        if not exists(path_to_video):
            print("File does not exists")
            return
        
        if not isinstance(path_to_video,str):
            path_to_video = str(path_to_video)

        video = cv2.VideoCapture(path_to_video)
        self.cfg_vid_nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cfg_vid_fps = int(video.get(cv2.CAP_PROP_FPS))
        self.cfg_vid_size_x = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cfg_vid_size_y = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # prelocate output for speedup
        # accuracy of mediapipe detection
        self.mp_accuracy = np.empty([self.cfg_vid_nframes, self.cfg_mp_max_n_hands])
        self.mp_accuracy[:] = np.nan
        # 0x63 np array with normalised locations of xyz component of 21
        # tracked points from hand
        self.mp_positions_norm = np.empty(
            [self.cfg_vid_nframes, len(self.cfg_mp_labels), self.cfg_mp_max_n_hands]
        )
        self.mp_positions_world = np.empty(
            [self.cfg_vid_nframes, len(self.cfg_mp_labels), self.cfg_mp_max_n_hands]
        )
        self.mp_positions_norm[:] = np.nan
        self.mp_positions_world[:] = np.nan
        self.mp_hand_labels = []
        self.mp_hand_length = []

        # loop over all frames
        for idxf in tqdm(range(0, self.cfg_vid_nframes)):
            success, tmp_image = video.read()
            tmp_norm_landmarks, tmp_world_landmarks, tmp_acc, tmp_hand_label = find_handlandmarks( self, tmp_image=tmp_image)

            # append results in np array
            self.mp_accuracy[idxf, :] = tmp_acc
            self.mp_positions_norm[idxf, :, :] = tmp_norm_landmarks
            self.mp_positions_world[idxf, :, :] = tmp_world_landmarks
            self.mp_hand_labels.append(tmp_hand_label)

        print("MP analysis successful")

    def find_roi(self, thresh_hand_acc=0.5, label_method = 'MP'):
        """ Finds the area of the video in which hand(s) are located.


        Parameters
        ----------
        cfg_mp_roi : NaN
            Regions of interest per hand prelocated as NaN if not processed.

        Returns
        -------
        cfg_mp_roi : float
            x and y values (axis = 0) with min and max (axis = 1) for all hands (axis = 2)
        """

        if label_method != 'MP' and label_method != 'SIDE':
            print("Specifiy hand label method either 'MP' or 'SIDE'")
            return
        
        if not all(
            hand_lbs == self.mp_hand_labels[0] for hand_lbs in self.mp_hand_labels
        ):
            print("Hand assignment switched during MP analysis")

        lbs_hand = np.array(self.mp_hand_labels)
        lbs_hand_count = np.unique(lbs_hand, axis=0, return_counts=True)
        lbs_unique = np.unique(lbs_hand_count[0].flatten().tolist())
        single_hand = False

        if len(lbs_unique) == 1 and "" in lbs_unique:
            print("No hand labels found")
            self.cfg_mp_n_hands = 0
            return

        elif len(lbs_unique) == 2 and "" not in lbs_unique:
            print("Two hand labels found")
            single_hand = False
            self.cfg_mp_n_hands = 2

        elif len(lbs_unique) == 2 and "" in lbs_unique:
            print("Only one hand label found")
            single_hand = True
            self.cfg_mp_n_hands = 1

        elif len(lbs_unique) == 3 and "" in lbs_unique:
            print("Two hand labels found")
            single_hand = False
            self.cfg_mp_n_hands = 2

        idx_most_occ = np.argmax(lbs_hand_count[1])

        if lbs_hand_count[1][idx_most_occ] / lbs_hand_count[1].sum() > thresh_hand_acc:
            self.mp_main_hand_label = lbs_hand_count[0][idx_most_occ]
            print(
                "Hand assignment consistent in {} % of the cases".format(
                    round(lbs_hand_count[1][idx_most_occ] / lbs_hand_count[1].sum(), 2)
                    * 100
                )
            )

        else:
            self.mp_main_hand_label = np.array(["", ""])
            print(
                "Hand assignment consistent in {} % of the cases. No hands labels assigned".format(
                    round(lbs_hand_count[1][idx_most_occ] / lbs_hand_count[1].sum(), 2)
                    * 100
                )
            )

        # min and max per hand
        idx_cor_lbs = np.logical_and(
            self.mp_hand_labels == self.mp_main_hand_label,
            self.mp_accuracy > thresh_hand_acc,
        )

        # first hand
        self.cfg_mp_roi[0, 0, 0] = (
            np.nanmin(self.mp_positions_norm[idx_cor_lbs[:, 0], 0::3, 0])
            * self.cfg_vid_size_x
        )
        self.cfg_mp_roi[0, 1, 0] = (
            np.nanmax(self.mp_positions_norm[idx_cor_lbs[:, 0], 0::3, 0])
            * self.cfg_vid_size_x
        )
        self.cfg_mp_roi[1, 0, 0] = (
            np.nanmin(self.mp_positions_norm[idx_cor_lbs[:, 0], 1::3, 0])
            * self.cfg_vid_size_y
        )
        self.cfg_mp_roi[1, 1, 0] = (
            np.nanmax(self.mp_positions_norm[idx_cor_lbs[:, 0], 1::3, 0])
            * self.cfg_vid_size_y
        )

        if single_hand:
            return

        # second hand
        self.cfg_mp_roi[0, 0, 1] = (
            np.nanmin(self.mp_positions_norm[idx_cor_lbs[:, 1], 0::3, 1])
            * self.cfg_vid_size_x
        )
        self.cfg_mp_roi[0, 1, 1] = (
            np.nanmax(self.mp_positions_norm[idx_cor_lbs[:, 1], 0::3, 1])
            * self.cfg_vid_size_x
        )
        self.cfg_mp_roi[1, 0, 1] = (
            np.nanmin(self.mp_positions_norm[idx_cor_lbs[:, 1], 1::3, 1])
            * self.cfg_vid_size_y
        )
        self.cfg_mp_roi[1, 1, 1] = (
            np.nanmax(self.mp_positions_norm[idx_cor_lbs[:, 1], 1::3, 1])
            * self.cfg_vid_size_y
        )

        if label_method == 'SIDE' and single_hand:
            self.mp_hand_labels = np.repeat([self.mp_main_hand_label],self.cfg_vid_nframes,axis=0)
            print('All hand labels are forced to be main hand label')
            
        if label_method == 'SIDE' and not single_hand:
            if np.nanmedian(self.mp_positions_norm[:,0,0]) < np.nanmedian(self.mp_positions_norm[:,0,1]):
                self.mp_hand_labels = [['Left','Right']] * self.cfg_vid_nframes
            elif np.nanmedian(self.mp_positions_norm[:,0,0]) > np.nanmedian(self.mp_positions_norm[:,0,1]):
                self.mp_hand_labels = [['Right','Left']] * self.cfg_vid_nframes
            print('All hand labels are forced to be main hand labels')


            
    def show_video(self, path_to_video, write_video=False):
        """ This function displays a video.

        If ROI for hands have been found, this video shows the ROI per hand with labels

        Parameters
        ----------
        path_to_video : string
            Path to video to be displayed.

        Returns
        -------
        None.

        """

        ccs = [(231, 217, 0), (0, 138, 255)]
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cv_out = cv2.VideoWriter("test.avi", fourcc, 30.0, (1280, 720))

        # For video input:
        if not exists(path_to_video):
            print("File does not exists")

        video = cv2.VideoCapture(path_to_video)

        for f in range(0, self.cfg_vid_nframes // 2):
            success, image = video.read()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            if success:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                if not math.isnan(self.cfg_mp_roi[0, 0, 0]):
                    # Add ROI information if avaliable
                    image = cv2.rectangle(
                        image,
                        self.cfg_mp_roi.astype(int)[:, 0, 0],
                        self.cfg_mp_roi.astype(int)[:, 1, 0],
                        ccs[0],
                        3,
                    )
                    image = cv2.rectangle(
                        image,
                        self.cfg_mp_roi.astype(int)[:, 0, 1],
                        self.cfg_mp_roi.astype(int)[:, 1, 1],
                        ccs[1],
                        3,
                    )
                    # add legend

                    cv2.putText(
                        img=image,
                        text="ROI " + self.mp_main_hand_label[1],
                        org=(0, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=ccs[0],
                        thickness=1,
                    )

                    cv2.putText(
                        img=image,
                        text="ROI " + self.mp_main_hand_label[0],
                        org=(0, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.8,
                        color=ccs[1],
                        thickness=1,
                    )

                if write_video:
                    cv_out.write(image)

                cv2.imshow("MediaPipe Hands", image)
                if cv2.waitKey(5) & 0xFF == 27:  # esc key for exit
                    break
        cv_out.release()
        video.release()
        cv2.destroyAllWindows()

    def frequency_estimations(
            self, thresh_hand_acc = .5, filter_boundaries = [cfg_bandpass_fmin, cfg_bandpass_fmax], cfg_n_components = 3, freqs_oi = [cfg_frequency_win_oi_min, cfg_frequency_win_oi_max], to_plot=False, to_save=False, output_fname = 'output.png'
        ):
        """
        This function does a frequency decomposition based on all tracked points via a FFT and a PCA.

        Parameters
        ----------
        thresh_hand_acc : float64
            Threshold for confidence level of MP for a label to be used
        filter_boundaries : list, optional
            Cutoff frequencies for bandpass filter. The default is [2, 12].
        cfg_n_components : int, optional
            Number of components from PCA to be returned. The default is 3.
        freqs_oi : list, optional
            Frequency range in which tremor is to be detected. The default is [4, 8].
        to_plot : bool, optional
            Indicate if plotting should be done. The default is False.
        to_save : bool, optional
            Indicate if plot produced should be saved. The default is False.
        output_fname : str, optional
            Output file name for saving the plot. The default is 'output.png'.

        Returns
        -------
        tuple
            If self.cfg_mp_n_hands == 1:
                freqs_pca_h0 : array_like
                    Frequencies for the first hand.
                pc_spec_h0 : array_like
                    Power spectral density for the first hand.
                mp_main_hand_label : str
                    Label of the main hand.

            If self.cfg_mp_n_hands == 2:
                freqs_pca_h0 : array_like
                    Frequencies for the first hand.
                pc_spec_h0 : array_like
                    Power spectral density for the first hand.
                mp_main_hand_label : str
                    Label of the main hand.
                freqs_pca_h1 : array_like
                    Frequencies for the second hand.
                pc_spec_h1 : array_like
                    Power spectral density for the second hand.
        """
        # Function code here
        idx_start_label = 4
        # correct labels based on mediapipe accuracy and hand level assignment
        idx_cor_lbs = np.logical_or(
            self.mp_hand_labels == self.mp_main_hand_label,
            self.mp_accuracy > thresh_hand_acc,
        )
        # only one axis for now and first hand
        raw_h0 = pd.DataFrame(self.mp_positions_norm[:, idx_start_label:, 0], columns=self.cfg_mp_labels[idx_start_label:])

        # only one axis for now and first hand
        raw_h1 = pd.DataFrame(self.mp_positions_norm[:, idx_start_label:, 1], columns=self.cfg_mp_labels[idx_start_label:])

        raw_h0[np.invert(idx_cor_lbs[:, 0])] = np.nan
        raw_h1[np.invert(idx_cor_lbs[:, 1])] = np.nan

        raw_h0 = raw_h0 - raw_h0.mean()
        raw_h1 = raw_h1 - raw_h1.mean()

        interp_h0 = raw_h0.interpolate().fillna(method="bfill", axis=0)
        interp_h1 = raw_h1.interpolate().fillna(method="bfill", axis=0)

        raw_interp = np.stack((interp_h0, interp_h1), axis=2)
        print("Position interpolation successful")

        if not tuple(map(sum, zip(raw_interp.shape,(0,idx_start_label,0)))) == self.mp_positions_norm.shape and self.cfg_mp_n_hands == 2:
            print("Size after interpolation not correct")
            return
        elif np.isnan(raw_interp).any() and self.cfg_mp_n_hands == 2:
            print("Array still contains NaN value")
            return
        elif np.isnan(raw_interp[:, :, 0]).any() and self.cfg_mp_n_hands == 1:
            print("Array for first hand still contains NaN value")
            return

        if self.cfg_mp_n_hands == 2:
            print(
                "{} % detection accuracy in correct labeled frames".format(
                    round(np.nanmean(self.mp_accuracy[idx_cor_lbs]), 3) * 100
                )
            )
        elif self.cfg_mp_n_hands == 1:
            print(
                "{} % detection accuracy in correct labeled frames".format(
                    round(np.nanmean(self.mp_accuracy[idx_cor_lbs[:, 0], 0]), 3) * 100
                )
            )

        # initialize PCA 
        pca = PCA(n_components = cfg_n_components)
        pcs = np.empty((len(raw_interp), cfg_n_components, self.cfg_mp_max_n_hands))

        # prelocate vars and high-pass filter first hand
        filt_z = np.empty(raw_interp.shape)
        filt_z[:, :, 0] = butter_bandpass_filter(
            raw_interp[:, :, 0], filter_boundaries[0], filter_boundaries[1], self.cfg_vid_fps
        )

        # pca first hand
        pcs[:, :, 0] = pca.fit_transform(filt_z[:, :, 0])

        # filter each component
        freqs_pca_h0, specs_pca_h0 = signal.welch(
            pcs[:, :, 0], 
            nperseg = 5 * self.cfg_vid_fps, 
            fs = self.cfg_vid_fps, 
            detrend='linear',
            scaling="spectrum",
            axis=0
        )  

        # filter and pca second hand
        if self.cfg_mp_n_hands == 2:
            filt_z[:, :, 1] = butter_bandpass_filter(
                raw_interp[:, :, 1], filter_boundaries[0], filter_boundaries[1], self.cfg_vid_fps
            )
            pcs[:, :, 1] = pca.fit_transform(filt_z[:, :, 1])
            freqs_pca_h1, specs_pca_h1 = signal.welch(
                pcs[:, :, 1], 
                nperseg = 5 * self.cfg_vid_fps, 
                fs = self.cfg_vid_fps, 
                detrend='linear',
                scaling="spectrum",
                axis=0
            )  

            
        # colors
        ccs_raw = plt.cm.viridis(np.linspace(0.2, 0.9, raw_interp.shape[1]))
        ccs_pca = plt.cm.viridis(np.linspace(0.2, 0.9, cfg_n_components))

        # time vector for plotting
        t_vec = np.linspace(
            0, self.cfg_vid_nframes / self.cfg_vid_fps, self.cfg_vid_nframes
        )

        pc_spec_h0, peaks, eigen_ratio = pcs2spec(cfg_n_components=cfg_n_components, cfg_freqs_oi=freqs_oi, specs=specs_pca_h0, freqs=freqs_pca_h0)
        
        if self.cfg_mp_n_hands == 2:
            pc_spec_h1, peaks, eigen_ratio = pcs2spec(cfg_n_components=cfg_n_components, cfg_freqs_oi=freqs_oi, specs=specs_pca_h1, freqs=freqs_pca_h1)

        print(len(pc_spec_h0))

        if to_plot and self.cfg_mp_n_hands == 2:
            plt.figure(figsize=(8, 5),dpi=300)

            plt.subplot(3, 2, 1)
            plt.title("Raw for {} hand".format(self.mp_main_hand_label[1]))
            for sig, color in zip(filt_z[:, :, 0].T, ccs_raw):
                plt.plot(t_vec, sig, color=color, linewidth=0.5, alpha=0.5)

            plt.subplot(3, 2, 3)
            plt.title("PCA")
            for sig, color in zip(pcs[:, :, 0].T, ccs_pca):
                plt.plot(t_vec[:], sig, color=color)
            plt.xlabel("Time [s]")


            plt.subplot(3, 2, 5)
            plt.title("Spectra")
            plt.plot(
                freqs_pca_h0, pc_spec_h0, color=ccs_pca[0, :], label="PCA"
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD")
            plt.xlim(2, 12)
            plt.ylim(0,np.nanmax(np.sqrt(specs_pca_h0)))

            plt.subplot(3, 2, 2)
            plt.title("Raw for {} hand".format(self.mp_main_hand_label[0]))
            for sig, color in zip(filt_z[:, :, 1].T, ccs_raw):
                plt.plot(t_vec, sig, color=color, linewidth=0.5, alpha=0.5)

            plt.subplot(3, 2, 4)
            plt.title("PCA")
            for sig, color in zip(pcs[:, :, 1].T, ccs_pca):
                plt.plot(t_vec, sig, color=color)
            plt.xlabel("Time [s]")

            plt.subplot(3, 2, 6)
            plt.title("Spectra")
            plt.plot(
                freqs_pca_h1, pc_spec_h1, color=ccs_pca[0, :], label="PCA"
            )

            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD")
            plt.xlim(freqs_oi)

            plt.tight_layout()



        if to_plot and self.cfg_mp_n_hands == 1:
            plt.figure(figsize=(8, 5))

            plt.subplot(3, 1, 1)
            plt.title("Raw for {} hand".format(self.mp_main_hand_label[0]))
            for sig, color in zip(filt_z[:, :, 0].T, ccs_raw):
                plt.plot(t_vec, sig, color=color, linewidth=0.5, alpha=0.5)

            plt.subplot(3, 1, 2)
            plt.title("PCs")
            for sig, color in zip(pcs[:, :, 0].T, ccs_pca):
                plt.plot(t_vec, sig, color=color)
            plt.xlabel("Time [s]")

            plt.subplot(3, 1, 3)
            plt.title("Spectra")
            plt.plot(
                freqs_pca_h0, pc_spec_h0, color=ccs_pca[0, :], label="PCA"
            )
            

            plt.vlines(
                freqs_pca_h0[np.argmax(pc_spec_h0)],
                0,
                np.nanmax(np.sqrt(pc_spec_h0)),
                color=[0, 0, 0],
                label="Peak at {}Hz".format(
                    round(freqs_pca_h0[np.argmax(pc_spec_h0)], 2)
                ),
                linestyles="dotted",
            )

            plt.legend(frameon=False, loc=1)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD")
            plt.xlim(freqs_oi)
            plt.ylim([0,np.max(pc_spec_h0)])
            plt.tight_layout()
            #plt.show()


        if to_save:
            plt.gcf()
            fname = output_fname
            plt.savefig(fname, dpi=300)
            plt.close('all')

        if self.cfg_mp_n_hands == 1:
            return freqs_pca_h0, pc_spec_h0, self.mp_main_hand_label

        elif self.cfg_mp_n_hands == 2:
            return freqs_pca_h0, pc_spec_h0, self.mp_main_hand_label, freqs_pca_h1, pc_spec_h1

    def conservative_handlabels(self):  # function to determine handedness via the x axis position in the image.
        hand0=np.array(self.mp_positions_norm[:,(9*3):((9*3))+3,0]) #get position of point 9 of each hand for each frame.
        hand1=np.array(self.mp_positions_norm[:,(9*3):((9*3))+3,1])

        for k in range(0,len(hand0)): #iterate over every frame
            if hand0<hand1:     #if the position of the point is smaller then the other, it is more right on the image.
                self.mp_hand_labels[k][0]='Right'   #the labels are reset, accordingly.
                self.mp_hand_labels[k][1]='Left'    #the labels are reset, accordingly.
            else:
                self.mp_hand_labels[k][1]='Left'    #the labels are reset, accordingly.
                self.mp_hand_labels[k][0]='Right'   #the labels are reset, accordingly.

            if self.mp_hand_labels[k][0]=='Right':   #the right hand will be stored in the first axis, the left in the second axis of the array with shape 
                                                     # [framelenght,x/y/z of 21 points, righthand/lefthand] [i,63,2]
                self.mp_positions_norm[k,:,0]=self.mp_positions_norm[k,:,0] 
                self.mp_positions_norm[k,:,1]=self.mp_positions_norm[k,:,1]
            else:
                self.mp_positions_norm[k,:,1]=self.mp_positions_norm[k,:,0]
                self.mp_positions_norm[k,:,0]=self.mp_positions_norm[k,:,1]


    def get_mp_hand_lengths(self):
        hand0_point0 = np.array(self.mp_positions_norm[:,(0*3):((0*3))+3,0])
        hand1_point0 = np.array(self.mp_positions_norm[:,(0*3):((0*3))+3,1])
        hand0_point12 = np.array(self.mp_positions_norm[:,(12*3):((12*3))+3,0])
        hand1_point12 = np.array(self.mp_positions_norm[:,(12*3):((12*3))+3,1] )

        hand_length0 = np.median(np.linalg.norm(hand0_point0-hand0_point12,axis=1)) #calculate distance between point 9 and 12 fpr each 
                                                                        #hand  in 3d space over time and take the median (more robust to outliers.)

        hand_length1 = np.median(np.linalg.norm(hand1_point0-hand1_point12,axis=1 ) )

        self.mp_hand_lengths = np.array([hand_length0,hand_length1])
