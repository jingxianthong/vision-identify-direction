# 🧭 Raspbot Arrow Navigation: System Analysis & Edge Case Failure

## System Showcase
[![Raspbot System Showcase](https://img.youtube.com/vi/zwyWs2UIMFQ/0.jpg)](https://youtu.be/zwyWs2UIMFQ)
*(Click the image above to watch the video demonstration)*

## 1. Expected System Behavior
The Arrow Navigation system is designed to act as an autonomous visual-motor pipeline. Under optimal conditions, the expected workflow is:
1. **Visual Input:** The Raspberry Pi camera captures a live video frame.
2. **AI Inference:** The custom-trained YOLOv8 model (`best.pt`) analyzes the frame to detect specific directional classes (`Left`, `Right`, `Forward`).
3. **State Memory:** The system registers the detected class and locks it into memory to prevent motor stuttering between processing frames.
4. **Motor Execution:** The robot translates the active state into continuous Mecanum wheel movement (e.g., rotating left when a "Left" arrow is identified) until the visual cue is removed or changed.

## 2. The Problem Faced (Edge Case)
Despite the AI model being trained on a custom dataset of over 100 images of left and right arrows, a critical failure occurred during live testing. 

### Visual Examples of the Failure State

| Left Arrow Edge Case | Right Arrow Edge Case |
| :---: | :---: |
| <img width="640" height="480" alt="arrow_0" src="https://github.com/user-attachments/assets/7629038b-2452-45dc-84c2-4c2808d5558f" /> | <img width="640" height="480" alt="arrow_2" src="https://github.com/user-attachments/assets/980275ed-ce3d-4b9c-a52a-1eedaf84ce36" /> |

When presented with a hand-drawn arrow on a blank white sheet of paper, the system registered **zero detections**. The camera functioned perfectly, and the environment was well-lit, yet the AI was completely blind to the drawing. The specific drawing used for this test was a large arrow outlined in blue pen, with the interior "colored in" using diagonal sketch lines (hatching), leaving significant white space visible inside the lines of the arrow.

## 3. How Machine Learning Actually "Sees"
To understand this failure, it is necessary to understand how a Convolutional Neural Network (CNN) like YOLOv8 processes visual data. 

When a human looks at a drawing, they understand the abstract *concept* of an arrow—a geometric shape consisting of a rectangle attached to a triangle. Machine learning models **do not understand concepts**. They are purely mathematical pattern-matching engines. 

During the training phase (when the 100 images were fed into Colab), the AI did not learn what an "arrow" is. Instead, it learned to look for specific clusters of pixels, gradients, and edge contrasts. 
* If the training dataset consisted mostly of printed signs, solid digital graphics, or heavily colored-in markers, the AI learned a rule: *"A 'Right Arrow' is a solid, continuous block of dark pixels arranged in a specific shape against a contrasting background."*
* It applies mathematical filters over the live camera feed, searching for that exact density and arrangement of pixels.

## 4. Diagnosis: Why the System Failed
The system failed to detect the hand-drawn arrow because the physical drawing fundamentally violated the mathematical pixel rules the AI learned during training.

1. **The Texture Mismatch:** Because the arrow was shaded using diagonal pen strokes (hatching) rather than a solid fill, the AI's filters did not see a solid block of color. Instead, it saw thin blue lines with massive amounts of white background bleeding through the interior of the shape. Because this "striped" pixel pattern did not match the "solid" pixel pattern of the 100 training images, the AI mathematically rejected it as a non-target object.
2. **The Background Ratio:** The vast expanse of the blank white paper compared to the size of the sketched lines may have disrupted the bounding box scaling. The AI calculates contrast edges; the fragmented edges of the sketchy pen lines against the stark white background diluted the confidence score below the acceptable threshold to trigger a detection. 

In summary, the AI is not broken; it is functioning exactly as trained. It successfully identified that the pixel pattern of a "hatched sketch" is not statistically similar enough to the pixel pattern of the "solid arrows" it was taught to look for.
