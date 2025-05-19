from abc import ABC, abstractmethod


class ImageSearchRepository(ABC):

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def connectDB(self):
        pass

    @abstractmethod
    def image_embedding(self,image_path):
        pass

    @abstractmethod
    def search(self,query_vec,top_k):
        pass