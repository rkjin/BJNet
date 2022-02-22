# CFNet modified  
## Better Job net

BJNet_1st
  1. fused cost volume part & cascade cost volume part & hourglass 제거
  2. fused cost volume 중 gw4 만 사용
  3. prediction time and memory was decreased less than 2/3.
  
BJNet_2nd
  1. fused cost volume part & cascade volume part & hourglass 제거
  2. fused volume 중 gw3 만 사용
  3. 1st에 비하여 error 감소, 변환시간 증가

BJNet_3rd
  1. fused cost volume part & cascade volume part & hourglass 제거
  2. disparity variance modified<br>
     variance = sum{(d*sigma(di) - sum(d*sigma(di)))^2} / (N-1)
  3. new cost volume generation<br>
     new cost = 1/abs(feature_left - feature_right)<br>
     gwc & concat cost volume generation 제거

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
