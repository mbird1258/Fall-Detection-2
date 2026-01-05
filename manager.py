import utils
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from datetime import datetime
import pickle


class CameraManager:
    def __init__(self, 
                 cap: cv2.VideoCapture,
                 CamViewDepth, 
                 NetworkManager: utils.NetworkManager, 
                 OutputDirectoryName=None,
                 SaveVideoOfIncident=True, 
                 IncidentVideoLength=[2000,1000], 
                 TimeBeforeCullBody=250,
                 VelocityThreshold=0.05,
                 CooldownBetweenIncidents=1000,
                 HeadYThresh = 0.5,
                 HeadVThresh = 0.003,
                 KneeYThresh = 0.5,
                 KneeVThresh = 0.0005,
                 SaveGraphs = False):
        '''
        args:
            cap: cv2 videocapture of camera/video
            CamViewDepth: Distance between camera and the plane in front of the camera that on which the image lies (distance to image plane) (pixels)
            OutputDirectoryName: output directory name
            SaveVideoOfIncident: To save video of every incident or not
            IncidentVideoLength: If saving videos, how long before and after incident to save (milliseconds)
            TimeBeforeCullBody: How long of body not being detected before body is assumed to be gone (milliseconds)
            VelocityThreshold: How big a median distance between bodies' corresponding joints in 2 frames before determining body in frame 2 is not same (m/s)
            CooldownBetweenIncidents: How long before another incident is reported (avoids 50 incidents reported in one fall) (milliseconds)
        '''

        self.cap = cap
        self.OutputDirectoryName = OutputDirectoryName if SaveVideoOfIncident==True else None
        path = f"Out/{self.OutputDirectoryName}/"
        if not os.path.exists(path): os.makedirs(path)

        self.SaveVideoOfIncident = SaveVideoOfIncident
        self.IncidentVideoLength = IncidentVideoLength if SaveVideoOfIncident==True else None
        self.TimeBeforeCullBody = TimeBeforeCullBody
        self.VelocityThreshold = VelocityThreshold
        self.CooldownBetweenIncidents = CooldownBetweenIncidents

        self.AsyncUpdateVideoTimes = [] if SaveVideoOfIncident==True else None
        self.VideoNum = [0, 0] if SaveVideoOfIncident==True else None
        self.IncidentVideo = [] if SaveVideoOfIncident==True else None
        self.bodies = []
        self.log = log()
        self.PoseManager = utils.PoseManager(CamViewDepth)
        self.NetworkManager = NetworkManager

        self.HeadYThresh = HeadYThresh
        self.HeadVThresh = HeadVThresh
        self.KneeYThresh = KneeYThresh
        self.KneeVThresh = KneeVThresh
        
        self.SaveGraphs = SaveGraphs

        self.t_init = self.t0 = datetime.now().timestamp()

    def UpdateBodies(self, img, FrameMS):
        '''
        method:
            get median distance between each bodies' joints
            if distance < thresh:
                bodies are matched
            else:
                body2 ≠ body1, create body2 as new body object
        
        returns:
            [body1, body2, ...]
            each entry(body) is an object containing the xyz, velocity, and acceleration of each joint
        '''
        
        if len(self.bodies) == 0:
            _, BodyCoordinates2 = self.PoseManager.GetBodyPose(img, FrameMS)
            for BodyCoords in BodyCoordinates2:
                self.bodies.append(body(BodyCoords, FrameMS))
            return
        
        BodyCoordinates1 = np.array([body.xyz for body in self.bodies]) # shape: [# bodies 1, # joints, 3]
        _, BodyCoordinates2 = self.PoseManager.GetBodyPose(img, FrameMS) # shape: [# bodies 2, # joints, 3]
        
        if len(BodyCoordinates2) == 0:
            return
        
        TimeLastUpdated1 = np.array([body.TimeLastUpdated for body in self.bodies]) # shape: [# bodies 1]
        dtArr = FrameMS - TimeLastUpdated1 # shape: [# bodies 1]

        VelocityArr = (BodyCoordinates1[:, np.newaxis, :, :] - BodyCoordinates2[np.newaxis, :, :, :])/dtArr[:, np.newaxis, np.newaxis, np.newaxis] # shape: [# bodies1, # bodies2, # joints, 3]
        MatchArr1 = np.nanmax(np.sqrt(np.sum(VelocityArr**2, axis=3)), axis=2)
        MatchArr2 = np.copy(MatchArr1)

        while True:
            ind1, ind2 = np.unravel_index(np.nanargmin(MatchArr1, axis=None), MatchArr1.shape)
            vel = VelocityArr[ind1, ind2]

            if MatchArr1[ind1, ind2] > self.VelocityThreshold:
                break

            self.bodies[ind1].UpdateData(BodyCoordinates2[ind2], vel, FrameMS)
            MatchArr1[ind1] = np.inf
            MatchArr1[:, ind2] = np.inf
            MatchArr2[:, ind2] = np.inf
        
        while True:
            ind1, ind2 = np.unravel_index(np.nanargmin(MatchArr2, axis=None), MatchArr2.shape)
            vel = VelocityArr[ind1, ind2]

            if MatchArr2[ind1, ind2] == np.inf:
                break

            self.bodies.append(body(BodyCoordinates2[ind2], FrameMS))
            MatchArr2[:, ind2] = np.inf

    def UpdateBodiesList(self, FrameMS):
        bodies = []

        for body in self.bodies:
            if FrameMS - body.TimeLastUpdated <= self.TimeBeforeCullBody:
                bodies.append(body)
            else:
                if self.SaveGraphs: body.SaveGraphs()
        
        self.bodies = bodies

    def DetectIncident(self, FrameMS, verbose=True):
        FallsOccured = 0

        for body in self.bodies:
            if not body.y_head:
                continue

            if FrameMS - body.LastIncidentTime < self.CooldownBetweenIncidents:
                continue

            if not ((body.y_head  < self.HeadYThresh*body.y_head_max and body.v_head  > self.HeadVThresh) or
                    (body.y_knee  < self.KneeYThresh*body.y_knee_max and body.v_knee  > self.KneeVThresh) or
                    (body.y_knee2 < self.KneeYThresh*body.y_knee_max and body.v_knee2 > self.KneeVThresh)):
                continue
            
            body.UpdateIncident(FrameMS)
            FallsOccured += 1
            
            if self.SaveVideoOfIncident:
                self.AsyncUpdateVideoTimes.append(FrameMS + self.IncidentVideoLength[1])
                if (FallsOccured == 1): 
                    self.VideoNum[0] += 1
            
            # ==!== Other things to do if an incident is detected ==!== #
            self.NetworkManager.send()

            if verbose:
                print(f"Time(s): {np.round(FrameMS/1000, 1)} - fall >.<")
        
        if self.SaveVideoOfIncident:
            return FallsOccured
        
        return FallsOccured

    def StoreVid(self, path, video):
        height, width, _ = video[0].shape

        out = cv2.VideoWriter(path, 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              self.cap.get(cv2.CAP_PROP_FPS), 
                              (width,height))

        for img in video:
            out.write(img)

        out.release()

    def UpdateVideoAsync(self, FrameMS, force=False):
        if force:
            for _ in self.AsyncUpdateVideoTimes:
                path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[1]}.mp4"
                self.StoreVid(path, self.IncidentVideo)

                self.VideoNum[1] += 1
            
            self.AsyncUpdateVideoTimes = []

            return
        
        mask = np.ones_like(self.AsyncUpdateVideoTimes).astype(np.bool_)

        for ind, time in enumerate(self.AsyncUpdateVideoTimes):
            if FrameMS > time:
                path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[1]}.mp4"
                self.StoreVid(path, self.IncidentVideo)

                self.VideoNum[1] += 1
                mask[ind] = False
                continue
        
        self.AsyncUpdateVideoTimes = np.array(self.AsyncUpdateVideoTimes)[mask].tolist()

    def main(self, save=False, TestFile = False):
        '''
        returns log object of incident data
        '''

        ## DEBUG
        if TestFile and not save:
            if not hasattr(self, 'playback_data'):
                with open(TestFile, 'rb') as f:
                    self.playback_data = pickle.load(f)
                self.playback_index = 0
                self.playback_length = len(self.playback_data)
            
            if self.playback_index >= self.playback_length:
                if self.SaveVideoOfIncident:
                    self.UpdateVideoAsync(None, True)
                    for body in self.bodies:
                        if self.SaveGraphs: body.SaveGraphs()
                return False, self.log
                
            frame_data = self.playback_data[self.playback_index]
            img = frame_data['img']
            dt = frame_data['dt']
            FrameMS = frame_data['framems']
            self.playback_index += 1
            # print(dt)
            # print(FrameMS)
            cv2.imshow("img", img)
            cv2.waitKey(0)
        else:
            # Case 2 & 3: Normal camera operation (with optional recording)
            res, img = self.cap.read()
            if not res:
                if self.SaveVideoOfIncident:
                    # If video terminates, save video of incident as is even if not full n milliseconds after passed
                    self.UpdateVideoAsync(None, True)
                    for body in self.bodies:
                        if self.SaveGraphs: body.SaveGraphs()
                return False, self.log
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t1 = datetime.now().timestamp()
            dt = t1 - self.t0
            FrameMS = (t1 - self.t_init) * 1000
            
            # Store frame data if saving
            if save and TestFile:
                if not hasattr(self, 'recorded_data'):
                    self.recorded_data = []
                self.recorded_data.append({
                    'img': img.copy(),
                    'dt': dt,
                    'framems': FrameMS
                })
        
        ## Standard
        '''res, img = self.cap.read()
        if not res:
            if self.SaveVideoOfIncident:
                # If video terminates, save video of incident as is even if not full n milliseconds after passed
                self.UpdateVideoAsync(None, True)
            return False, self.log
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t1 = datetime.now().timestamp()
        dt = t1-self.t0
        FrameMS = (t1-self.t_init)*1000'''
        
        # If saving video of incidents, keep a running video of the past n milliseconds to save when an incident occurs
        if self.SaveVideoOfIncident:
            self.IncidentVideo.append(img)
            self.IncidentVideo = self.IncidentVideo[-int(self.IncidentVideoLength[0]/1000/dt):] # dt = seconds per frames, 1/dt = fps
        
        self.UpdateBodies(img, FrameMS) # match bodies, new unmatched bodies are assigned to new body objects
        self.UpdateBodiesList(FrameMS) # remove old unmatched bodies

        NumFalls = self.DetectIncident(FrameMS)
        if self.SaveVideoOfIncident:
            path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[0]}.mp4"
            self.log(FrameMS, NumFalls, path)
        else:
            self.log(FrameMS, NumFalls)

        if self.SaveVideoOfIncident:
            self.UpdateVideoAsync(FrameMS)
        
        if not TestFile or save: self.t0 = t1 # >>> NOT debug <<<
        
        return True, self.log

    def save(self, TestFile):
        with open(TestFile, 'wb') as f:
            pickle.dump(self.recorded_data, f)

class body:
    def __init__(self, xyz, FrameMS):
        # avg(11, 12)
        # avg(23, 24)
        # 26 28 30 || 25 27 29

        self.xyz = xyz # shape: [# joints, 3]

        self.y_knee_max = None
        self.y_head_max = None
        self.y_head = None
        self.y_knee = None
        self.y_knee2 = None
        self.v_head = None
        self.v_knee = None
        self.v_knee2 = None

        self.TimeLastUpdated = FrameMS
        self.LastIncidentTime = -np.inf

        self.history = {
            'time': [],
            'y_head': [],
            'y_head_max': [],
            'v_head': [],
            'y_knee': [],
            'y_knee2': [],
            'y_knee_max': [],
            'v_knee': [],
            'v_knee2': []
        }

    def UpdateData(self, xyz, vel, FrameMS):
        xyz = xyz.copy()
        xyz[:, 1] *= -1

        self.TimeLastUpdated = FrameMS

        if np.any(np.isnan(vel[[11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]])): # Require sight of these keypoints (shoulders hips knees ankles heels toes)
            self.y_knee_max = None
            self.y_head_max = None
            self.y_head = None
            self.y_knee = None
            self.y_knee2 = None
            self.v_head = None
            self.v_knee = None
            self.v_knee2 = None

            return

        head = (xyz[11]+xyz[12])/2
        crotch = (xyz[23]+xyz[24])/2
        hip = xyz[24]
        knee = xyz[26]
        ankle = xyz[28]
        heel = xyz[30]
        
        self.y_knee_max = np.linalg.norm(knee-ankle)+\
                          np.linalg.norm(ankle-heel)
        
        self.y_head_max = np.linalg.norm(head-crotch)+\
                          np.linalg.norm(hip-knee)+\
                          self.y_knee_max

        y_floor = min(xyz[:, 1])
        self.y_head = head[1]-y_floor
        self.y_knee = knee[1]-y_floor
        self.y_knee2 = xyz[25][1]-y_floor

        self.v_head = -(vel[11, 1]+vel[12, 1])/2 # np.linalg.norm((vel[11]+vel[12])/2)
        self.v_knee = -vel[26, 1] # np.linalg.norm(vel[26])
        self.v_knee2 = -vel[25, 1] # np.linalg.norm(vel[25])

        self.history['time'].append(self.TimeLastUpdated)
        self.history['y_head'].append(self.y_head)
        self.history['y_head_max'].append(self.y_head_max)
        self.history['v_head'].append(self.v_head)
        self.history['y_knee'].append(self.y_knee)
        self.history['y_knee2'].append(self.y_knee2)
        self.history['y_knee_max'].append(self.y_knee_max)
        self.history['v_knee'].append(self.v_knee)
        self.history['v_knee2'].append(self.v_knee2)
    
    def SaveGraphs(self):
        if not os.path.exists("DEBUG/"):
            os.makedirs("DEBUG/")

        dir = f"DEBUG/{len(os.listdir('DEBUG/'))}"

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        time_seconds = [t/1000 for t in self.history['time']]  # Convert to seconds

        # <==!==> <==!==> <==!==>
        # for i in range(len(self.history['y_knee_max'])):
        #     if self.history['y_knee_max'][i] < 0.4:
        #         self.history['y_knee_max'][i] = 0.5

        # for i in range(len(self.history['y_head_max'])):
        #     if self.history['y_head_max'][i] < 1.2:
        #         self.history['y_head_max'][i] = 1.3
        
        # Plot 1: y_head & y_head_max
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['y_head'], label='y_head', linewidth=2)
        plt.plot(time_seconds, self.history['y_head_max'], label='y_head_max', linewidth=2)
        plt.plot(time_seconds, np.array(self.history['y_head_max'])*0.5, label='Height Threshold', linewidth=2, c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title(f'Head Position vs Time - Body')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/head_position.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: y_knee & y_knee_max
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['y_knee'], label='y_knee', linewidth=2)
        plt.plot(time_seconds, self.history['y_knee_max'], label='y_knee_max', linewidth=2)
        plt.plot(time_seconds, np.array(self.history['y_knee_max'])*0.5, label='Height Threshold', linewidth=2, c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title(f'Knee Position vs Time - Body')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/knee_position.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: y_knee2 & y_knee_max
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['y_knee2'], label='y_knee2', linewidth=2)
        plt.plot(time_seconds, self.history['y_knee_max'], label='y_knee_max', linewidth=2)
        plt.plot(time_seconds, np.array(self.history['y_knee_max'])*0.5, label='Height Threshold', linewidth=2, c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title(f'Knee2 Position vs Time - Body')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/knee2_position.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['v_knee'], linewidth=2, label='v_knee')
        plt.plot(time_seconds, [0.0005]*len(time_seconds), linewidth=2, label='Velocity Threshold', c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('v_knee')
        plt.title(f'v_knee vs Time - Body')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/v_knee.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['v_knee2'], linewidth=2, label='v_knee2')
        plt.plot(time_seconds, [0.0005]*len(time_seconds), linewidth=2, label='Velocity Threshold', c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('v_knee2')
        plt.title(f'v_knee2 vs Time - Body')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/v_knee2.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_seconds, self.history['v_head'], linewidth=2, label='v_head')
        plt.plot(time_seconds, [0.003]*len(time_seconds), linewidth=2, label='Velocity Threshold', c='red')
        plt.xlabel('Time (s)')
        plt.ylabel('v_head')
        plt.title(f'v_head vs Time - Body')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{dir}/v_head.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Individual plots for each variable
        # variables = ['y_head', 'y_head_max', 'v_head', 'y_knee', 'y_knee2', 'y_knee_max', 'v_knee', 'v_knee2']
        # for var in variables:
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(time_seconds, self.history[var], linewidth=2)
        #     plt.xlabel('Time (s)')
        #     plt.ylabel(var)
        #     plt.title(f'{var} vs Time - Body')
        #     plt.grid(True, alpha=0.3)
        #     plt.savefig(f"{dir}/{var}.png", dpi=150, bbox_inches='tight')
        #     plt.close()

    def UpdateIncident(self, FrameMS):
        self.LastIncidentTime = FrameMS


class log:
    def __init__(self):
        self.contents = []
    
    def __call__(self, FrameMS, NumFalls, VideoPath=False):
        if NumFalls == 0:
            return

        if VideoPath:
            self.contents.append((FrameMS, NumFalls, VideoPath))
            return
        
        self.contents.append((FrameMS, NumFalls))
    
    def AccessVideoPathByInd(self, ind):
        return self.contents[ind][2]
    
    def __str__(self):
        if len(self.contents) == 0:
            return "No Contents"
        
        out = ""
        for ind, entry in enumerate(self.contents):
            out = out+f"\nIndex: {ind} || Time(s): {np.round(entry[0]/1000, 2)} || Num of falls: {entry[1]} || Recorded Videos: {len(entry) == 3}"
        
        return out
