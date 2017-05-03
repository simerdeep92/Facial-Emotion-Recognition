import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("source_emotion/*") #Returns a list of all folders with participant numbers

neutral_subjects = set()

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    print "part" + part
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        print "sessions --- " + sessions
        print (glob.glob("%s/*" %sessions))
        for files in glob.glob("%s/*" %sessions):
            current_session = sessions[-3:]
            print "cc----" + current_session
            file = open(files, 'r')            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.            
            emotion_files = sorted(glob.glob("source_images/%s/%s/*" %(part, current_session)))
            sourcefile_emotion = emotion_files[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = emotion_files[0] #do same for neutral image
            neutral_file = sourcefile_neutral.split("/")[-1]
            emotion_file = sourcefile_emotion.split("/")[-1]
            dest_neut = "sorted_set/neutral/%s" %neutral_file #Generate path to put neutral image
            dest_emot = "sorted_set/%s/%s" %(emotions[emotion], emotion_file) #Do same for emotion containing image
            if part not in neutral_subjects:
                copyfile(sourcefile_neutral, dest_neut) #Copy file
                neutral_subjects.union([part])
            copyfile(sourcefile_emotion, dest_emot) #Copy file
            