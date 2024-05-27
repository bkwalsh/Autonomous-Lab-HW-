import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm


def rotate_image(image, angle):
    '''
    Rotates a fruit image by a particular angle (in degrees)
    Parameters
    ----------
        image (cv2.Image or ndarray): Input image to be rotated
        angle (float): Angle of rotation, in degrees
    Returns
    -------
        result (cv2.Image): Rotated image, of shape (X,Y,3).
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def clip_edges(fruit_image,x_loc,y_loc,fruit_size=100,x_len=512,y_len=1024):
    '''
    Clip the edges of a fruit image to fit with background image boundaries.
    Parameters
    ----------
        fruit_image (cv2.Image or ndarray): Initial fruit image to be cropped
        x_loc (int): X-axis location of the fruit image
        y_loc (int): Y-axis location of the fruit image
        fruit_size (int): X- and Y-length of the fruit image
        x_len (int): X-length of the background image
        y_len (int): Y-length of the background image
    Returns
    -------
        fruit_image (cv2.Image or ndarray): Cropped version of the initial fruit image
    '''
    #X Coordinate Cutoffs
    #FROM TOP
    diff = x_loc - int(fruit_size/2)
    if diff < 0:
        fruit_image = fruit_image[-diff:,:,:]
    #FROM BOTTOM
    diff = x_len - (x_loc + int(fruit_size/2))
    if diff<0:
        fruit_image = fruit_image[:diff,:,:]
    #Y Coordinate Cutoffs
    #FROM TOP
    diff = y_loc - int(fruit_size/2)
    if diff < 0:
        fruit_image = fruit_image[:,-diff:,:]
    #FROM BOTTOM
    diff = y_len - (y_loc + int(fruit_size/2))
    if diff<0:
        fruit_image = fruit_image[:,:diff,:]
    #print('')
    return fruit_image


def save_video(save_vid_folder,video_name='out.mp4'):
    '''
    Save video from folder of slides using ffmpeg.
    Parameters
    ----------
        save_vid_folder (str): Location of folder containing image frames
        video_name (str): Name of the output .mp4 video
    Returns
    -------
        None
    '''
    os.popen(f"ffmpeg -framerate 30 -pattern_type glob -i '{save_vid_folder}/*.png' -c:v libx264 -pix_fmt yuv420p {video_name}")
    return

class FruitNinja:
    '''
    Play Fruit Ninja Game.
    '''
    def __init__(self):
        '''
        Initialize images, fruit trajectories, and crosshair controller.
        '''
        self.fruit_size = 100
        self.watermelon = cv2.resize(cv2.cvtColor(cv2.imread('fruit_imgs/watermelon.png'),\
                                            cv2.COLOR_BGR2RGB),(self.fruit_size,self.fruit_size))
        self.watermelon_mask = np.where(np.sum(self.watermelon,axis=2)==0,0,1)
        self.watermelon_mask = np.uint8(np.repeat(self.watermelon_mask[:,:,np.newaxis],3,axis=2))

        self.orange = cv2.resize(cv2.cvtColor(cv2.imread('fruit_imgs/orange.png'),\
                                        cv2.COLOR_BGR2RGB),(self.fruit_size,self.fruit_size))
        self.orange_mask = np.where(np.sum(self.orange,axis=2)==0,0,1)
        self.orange_mask = np.uint8(np.repeat(self.orange_mask[:,:,np.newaxis],3,axis=2))

        self.strawberry = cv2.resize(cv2.cvtColor(cv2.imread('fruit_imgs/strawberry.png'),\
                                            cv2.COLOR_BGR2RGB),(self.fruit_size,self.fruit_size))
        self.strawberry_mask = np.where(np.sum(self.strawberry,axis=2)==0,0,1)
        self.strawberry_mask = np.uint8(np.repeat(self.strawberry_mask[:,:,np.newaxis],3,axis=2))

        self.img_size = np.array([512,1024,3])
        self.blank_img = cv2.resize(cv2.cvtColor(cv2.imread('fruit_imgs/wood_background.png'),\
                                            cv2.COLOR_BGR2RGB),(self.img_size[1],self.img_size[0]))
        self.W_traj = []
        self.S_traj = []
        self.O_traj = []
        self.CH_controller_dict = {'type':'crosshairs','x':256,'y':256} #Start at 256,256
        
    
    def create_figure(self,object_position_list,show=False):
        '''
        Create a figure containing the objects listed in object_position_list, and return that final figure.
        Parameters
        ----------
            object_position_list (list): List of objects to be placed on top of the blank image.
            show (bool): If True, show the image.
        Returns
        -------
            blank_img (cv2.Image, or ndarray): Final image with objects (crosshairs, fruit) added.
        '''
        blank_img = self.blank_img.copy()
        for obj in object_position_list:
            region = blank_img[max(obj['x']-int(self.fruit_size/2),0):min(obj['x']+int(self.fruit_size/2),self.img_size[0]),\
                            max(obj['y']-int(self.fruit_size/2),0):min(obj['y']+int(self.fruit_size/2),self.img_size[1]),:]
            if obj['type']=='crosshairs': #Add crosshairs to final image
                cv2.line(blank_img,(obj['x']-20,obj['y']-20),(obj['x']+20,obj['y']+20),(255,255,255),6)
                cv2.line(blank_img,(obj['x']+20,obj['y']-20),(obj['x']-20,obj['y']+20),(255,255,255),6)
            else:
                if obj['type']=='watermelon': #Add watermelon, if object is present.
                    roto_watermelon = clip_edges(rotate_image(self.watermelon,obj['angle']),obj['x'],obj['y'])
                    roto_watermelon_mask = clip_edges(rotate_image(self.watermelon_mask,obj['angle']),obj['x'],obj['y'])
                    try: #For some reason there is still a boundary issue. So I just kill the fruit if it's giving me a headache.
                        region = roto_watermelon*roto_watermelon_mask + region*(1-roto_watermelon_mask)
                    except ValueError:
                        self.W_traj=[]
                        continue
                if obj['type']=='strawberry': #Add strawberry, if object is present.
                    roto_strawberry = clip_edges(rotate_image(self.strawberry,obj['angle']),obj['x'],obj['y'])
                    roto_strawberry_mask = clip_edges(rotate_image(self.strawberry_mask,obj['angle']),obj['x'],obj['y'])
                    try:
                        region = roto_strawberry*roto_strawberry_mask + region*(1-roto_strawberry_mask)
                    except ValueError:
                        self.S_traj = []
                        continue
                if obj['type']=='orange': #Add orange, if object is present.
                    roto_orange = clip_edges(rotate_image(self.orange,obj['angle']),obj['x'],obj['y'])
                    roto_orange_mask = clip_edges(rotate_image(self.orange_mask,obj['angle']),obj['x'],obj['y'])
                    try:
                        region = roto_orange*roto_orange_mask + region*(1-roto_orange_mask)
                    except ValueError:
                        self.O_traj = []
                        continue
                blank_img[max(obj['x']-int(self.fruit_size/2),0):min(obj['x']+int(self.fruit_size/2),self.img_size[0]),\
                    max(obj['y']-int(self.fruit_size/2),0):min(obj['y']+int(self.fruit_size/2),self.img_size[1]),:] = region
        blank_img+=np.uint8(np.random.normal(0,1,self.img_size))
        if show: #Show image, if applicable.
            plt.figure()
            plt.imshow(blank_img)
            plt.show()
        return blank_img

    def random_fruit_throw(self,speed=1500):
        '''
        Create a random trajectory across the background image with a fruit, using a parametric parabola equation.
        Parameters
        ----------
            speed (int): The number of points in the trajectory. Effectively, the larger the "speed", the slower the fruit.
        Returns
        -------
            x_coords (ndarray): List of x-coordinates for fruit trajectory
            y_coords (ndarray): List of y-coordinates for fruit trajectory
            roto_coords (ndarray): List of angles (based on constant rotational inertia) for fruit trajectory
        '''
        rotational_init = np.random.uniform(0,360)
        rotational_velocity = np.random.uniform(1,3) #degrees per frame
        a = np.random.uniform(5,40)
        b = np.int64(np.random.uniform(100,924))
        c = np.int64(np.random.uniform(100,312))
        if np.random.choice([True,False]):
            t_seq = np.linspace(-100,100,speed)
        else:
            t_seq = np.linspace(100,-100,speed)
        #Time-based parametric representation of a parabola
        full_coords = np.array([2*a*t_seq + b, a*(t_seq**2)+c]).T
        full_coords = full_coords[full_coords[:,0]>-self.fruit_size]
        full_coords = full_coords[full_coords[:,0]<self.img_size[1]+self.fruit_size]
        full_coords = full_coords[full_coords[:,1]>-self.fruit_size]
        full_coords = full_coords[full_coords[:,1]<self.img_size[0]+self.fruit_size]
        x_coords = np.int64(full_coords[:,1])
        y_coords = np.int64(full_coords[:,0])
        roto_coords = rotational_init + np.cumsum(rotational_velocity*np.ones(len(x_coords)))
        roto_coords = np.int64(roto_coords) % 360
        return x_coords,y_coords,roto_coords

    def check_fruit_collision(self,fruit_loc,min_dist_to_target=40):
        '''
        Determine whether crosshairs are within close enough proximity to a fruit to count as a collision
        Paramters
        ---------
            fruit_loc (ndarray): Location of the fruit in question
            min_dist_to_target (float): Distance after which crosshairs-fruit length constitutes a collision
        Returns
        -------
            is_collision (bool): True if the distance from crosshairs to fruit is less than minimum distance
        '''
        CH_loc = [self.CH_controller_dict['y'],self.CH_controller_dict['x']] # SWAPPED HERE
        fruit_dist = np.sqrt((CH_loc[0]-fruit_loc[0])**2 + (CH_loc[1]-fruit_loc[1])**2)
        is_collision = fruit_dist<min_dist_to_target
        return is_collision

    def play_game(self,crosshairs_controller,game_length=1000,save_vid_folder=None,max_controller_move=100):
        '''
        Main game playing function.
        Parameters
        ----------
            crosshairs_controller (function): Function that accepts crosshairs position [x,y] and the full
                image (cv2.Image) and returns a new crosshairs position [x_new,y_new].
            game_length (int): Number of frames to be used for the duration of the game
            save_vid_folder (str): Save each frame to this folder, then create a video (out.mp4) for the frames.
            max_controller_move (float): Maximum length of the controller move.
        Returns
        -------
            None
        '''
        if save_vid_folder is not None:
            if os.path.isdir(save_vid_folder):
                shutil.rmtree(save_vid_folder)
            os.mkdir(save_vid_folder)
        overall_score = 0
        for s_num in tqdm(range(game_length)):
            if len(self.W_traj)==0:
                self.W_traj = np.array(self.random_fruit_throw()).T
            if len(self.S_traj)==0:
                self.S_traj = np.array(self.random_fruit_throw()).T
            if len(self.O_traj)==0:
                self.O_traj = np.array(self.random_fruit_throw()).T
            obj_list = [self.CH_controller_dict,\
                {'type':'watermelon','x':self.W_traj[0][0],'y':self.W_traj[0][1],'angle':self.W_traj[0][2]},\
                {'type':'strawberry','x':self.S_traj[0][0],'y':self.S_traj[0][1],'angle':self.S_traj[0][2]},\
                {'type':'orange','x':self.O_traj[0][0],'y':self.O_traj[0][1],'angle':self.O_traj[0][2]}]
            #CHECK FOR COLLISION, WATERMELON
            killed_fruits = []
            if self.check_fruit_collision(self.W_traj[0]):
                self.W_traj=[]
                overall_score+=1
                killed_fruits.append('w')
            else:
                self.W_traj = self.W_traj[1:]
            #CHECK FOR COLLISION, STRAWBERRY
            if self.check_fruit_collision(self.S_traj[0]):
                self.S_traj=[]
                overall_score+=1
                killed_fruits.append('s')
            else:
                self.S_traj = self.S_traj[1:]
            #CHECK FOR COLLISION, ORANGE
            if self.check_fruit_collision(self.O_traj[0]):
                self.O_traj=[]
                overall_score+=1
                killed_fruits.append('o')
            else:
                self.O_traj = self.O_traj[1:]
                        
            #ADVANCE SLIDE
            new_slide = self.create_figure(obj_list)
            CH_loc = [self.CH_controller_dict['x'],self.CH_controller_dict['y']]

            new_crosshairs_position = np.int64(crosshairs_controller(CH_loc,new_slide,killed_fruits))
            CH_move_dist = np.sqrt((CH_loc[0]-new_crosshairs_position[0])**2 + \
                                   (CH_loc[1]-new_crosshairs_position[1])**2)
            if CH_move_dist > max_controller_move:
                raise ValueError(f'Movement distance exceeded {max_controller_move}')
            self.CH_controller_dict['x']=new_crosshairs_position[0]
            self.CH_controller_dict['y']=new_crosshairs_position[1]
            if save_vid_folder is not None:
                fname = os.path.join(save_vid_folder,f'slide_{str(s_num).zfill(5)}')
                plt.figure()
                plt.imshow(new_slide)
                plt.title(f'Slide Number: {s_num}, Score: {overall_score}')
                plt.savefig(fname)
                plt.close()
                plt.clf()
        if save_vid_folder is not None:
            if os.path.isfile('out.mp4'):
                os.remove('out.mp4')
            try:
                os.popen(f"ffmpeg -framerate 30 -pattern_type glob -i '{save_vid_folder}/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
            except:
                raise ValueError('Check whether ffmpeg is installed correctly. You can use "sudo apt install ffmpeg" to install.')
        return
    