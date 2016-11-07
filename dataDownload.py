from config.resources import video_resource
import youtube_dl
import os

class Folder:
    name=""
    url=""
    def __init__(self,name,url):
        self.name=name.lower()
        self.url=url

    def printline(self):
        print self.name
        print self.url
        return 0

    def download(self):
        if not os.path.exists(video_resource):
            os.makedirs(video_resource)
        prev=os.getcwd()
        if not os.path.exists(video_resource+self.name):
            os.mkdir(video_resource+self.name)
        path=video_resource+self.name
        os.chdir(path)
        print "youtube-dl -i "+self.url
        os.system("youtube-dl -i "+self.url)
        os.chdir(prev)
        return 0


if __name__=="__main__":
    horror=Folder("comedy","https://www.youtube.com/playlist?list=PLScC8g4bqD47QYmsgs_FBzcNvi3PPt2Ev")
    horror.printline()
    horror.download()

