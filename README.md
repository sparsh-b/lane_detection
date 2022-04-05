- Conversion to grayscale, binary thresholding, dilation & erosion, connected component analysis are applied in succession.
- Blobs of sensible sizes are isolated.
- Canny edges are detected.
- Hough lines are applied to detect the lines.
- The outputted lines are filtered to isolate 2 groups of lines which are sloped around 50 & 120 degrees.
- An average line which fits each group is calculated. This step gave 2 lines corresponding to 2 lanes.
- Once both the lanes are detected, the number of white pixels each of the lane lines sits on (when the lane lines are superimposed on the lanes) are calculated & the lane line sitting on fewer number of white pixels is considered as the broken line & the other as the solid line.
- [Link to output video](https://drive.google.com/file/d/14h9wH2Pq-qKO17Z9riQF8D00AI6523VX/view?usp=sharing)
- Usage `python3 detect_lane.py`

Note:
- The developed algorithm will work for straight lanes.
- It might have limitations when the lanes are not very bright.

