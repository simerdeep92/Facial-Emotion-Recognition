import glob
from shutil import copyfile
import csv
from operator import sub

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("Dataset/source_emotion/*") #Returns a list of all folders with participant numbers

neutral_subjects = []
sequential_dataset = []
relative_dataset = []


def Read_LandMark_file(filename):
    results = []
    with open(filename) as inputfile:
        for line in inputfile:
            results.extend(map(float,line.strip().split('   ')))
    return results

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    print "part" + part
    print ("%s/*" %x)
    for sessions in glob.glob("%s/*" %x) :#Store list of sessions for current participant
        print "sessions --- " + sessions
        #print (glob.glob("%s/*" %sessions))
        for files in glob.glob("%s/*" %sessions):
            current_session = sessions[-3:]
            print "cc----" + current_session
            file = open(files, 'r')            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.            
            #print("Landmarks/%s/%s/*" %(part, current_session))
            landmark_files = sorted(glob.glob("Dataset/Landmarks/%s/%s/*" %(part, current_session)))
            #print landmark_files
            sourcefile_emotion = landmark_files[-1] #get path for last image in sequence, which contains the emotion           
            sourcefile_neutral = landmark_files[0] #do same for neutral image
            neutral = Read_LandMark_file(sourcefile_neutral)
	        ##################Normal Relative Dataset ##############################
            emotion_data = Read_LandMark_file(sourcefile_emotion)
            relative = map(sub, emotion_data, neutral)
            temp = [sourcefile_emotion,(emotions[emotion])]
            temp.extend(relative)           
            relative_dataset.append(temp)

            i = 1
            ###### sequential dataset########
            for lndfile in landmark_files[1:]:
	            emotion_data = Read_LandMark_file(lndfile)
	            relative = map(sub, emotion_data, neutral)
	            temp = [sourcefile_emotion,(emotions[emotion] + str(i))]
	            temp.extend(relative)           
	            sequential_dataset.append(temp) 
	            i=i+1  
	        # if part not in neutral_subjects:
	        #     temp = [sourcefile_neutral,'neutral'] 
	        #     temp.extend(Read_LandMark_file(sourcefile_neutral))
	        #     relative_dataset.append(temp)
	        #     neutral_subjects.append(part)  




print(len(relative_dataset))
with open("landmark_data_relative.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(relative_dataset)

print(len(sequential_dataset))
with open("landmark_data_relative_sequential.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(sequential_dataset)