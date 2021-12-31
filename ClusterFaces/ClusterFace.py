import glob
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm




class ClusterFace():
    def __init__(self, dir_address='./', radius=0.2 ):
        self.extensions  = ['jpg','png','jpeg']
        self.dir_address = dir_address
        self.radius      = radius
        self.images_address = self.get_image_names(self.dir_address)
        self.embeddings  = self.get_embedding_faces(self.images_address)

    def get_image_names(self, address):
        files = []
        for extension in self.extensions:
            files.extend(glob.glob(os.path.join(self.dir_address,f'*.{extension}')))
        return files
        


    def cosine_similarity(self,vec1,vec2):
        cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
        return cos_sim
    def get_embedding_faces(self, images_address):
        pass



    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]
        
        while True:
            new_centroids = []
            new_centroids_data = {}
            for i in centroids:
                in_bandwidth = []
                in_bandwidth_idx = []
                centroid = centroids[i]
                for idx, featureset in enumerate(data):
                    if self.cosine_similarity(featureset,centroid) < self.radius:
                        in_bandwidth.append(featureset)
                        in_bandwidth_idx.append(idx)

                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))
                new_centroids_data[i] = in_bandwidth_idx

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids          = centroids
        self.new_centroids_data = new_centroids_data





if __name__ == "__main__":
    clsuerFace = ClusterFace()
    print (clsuerFace.images_name)