# CFNet modified  
## Better Job net

BJNet_1st
  1. fused cost volume & cascade coat volume & hourglass 제거
  2. fused volume 중 gw4 만 사용
  3. prediction time and memeory was decreased less than 2/3.
  
BJNet_2nd
  1. fused cost volume & cascade coat volume & hourglass 제거
  2. fused volume 중 gw3 만 사용

BJNet_3rd
  1. disparity variance modified<br>
     variance = sum{(d*sigma(di) - sum(d*sigma(di)))^2} / (N-1)
  3. new cost volume generation<br>
     new cost = 1/abs(feature_left - feature_right)
     

Calibrate Stereo camera<br>
  - look camera folder

Convert movie file<br>
  - $source ./scripts/movie.sh

Convert disparity map to color map 
  - look camera/disparity_to_3D.py

```
# Acknowledgements
Thanks to the CFNet.
Almost everything here stems from CFNet.
