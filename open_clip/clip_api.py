import numpy
import torch
from PIL import Image
import cv2

import open_clip.src as open_clip


class ClipDiscriminator:
    def __init__(self, prompts, device='cuda:2'):
        self.prompts = prompts
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
                                                                               pretrained='laion400m_e32')
        self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
        self.device = torch.device(device)

    def forward(self, image, decimal=4):
        """
        Zero Shot Classification
        Args:
            image: cv2 / Image / image_path
            decimal: digits number to save

        Returns: Probabilities of each class

        """
        if isinstance(image, numpy.ndarray):
            image = self.preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        elif isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0)
        elif isinstance(image, str):
            image = self.preprocess(Image.open(image)).unsqueeze(0)
        else:
            raise "Unknown Image Type!"

        text = self.tokenizer(self.prompts)

        image = image.to(self.device)
        text = text.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            results = text_probs.cpu().numpy().tolist()[0]
            results = [int(10 ** decimal * a) / 10 ** decimal for a in results]

            return results


if __name__ == '__main__':
    c = ClipDiscriminator(["a diagram", "a dog", "a cat"])
    # img = cv2.imread("CLIP.png")
    # img = Image.open("CLIP.png")
    img = 'CLIP.png'
    res = c.forward(img)
    print("Label probs:", res)
