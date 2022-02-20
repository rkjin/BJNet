# CFNet modified  
## BJNet stands for Better Job net

BJNet_fused_1st
  1. fused cost volume & cascade coat volume & hourglass 제거
  2. fused volume 중 gw4 만 사용
  3. prediction time and memeory was decreased less than 2/3.
  
BJNet_fused_2nd
  1. fused cost volume & cascade coat volume & hourglass 제거
  2. fused volume 중 gw3 만 사용

Calibrate Stereo camera<br>
  - look camera folder

Convert movie file<br>
  - $source ./scripts/movie.sh

Convert disparit map to color map 
  - look camera/disparity_to_3D.py

```
# Acknowledgements
Thanks to the CFNet.
Almost everything here stems from CFNet.
