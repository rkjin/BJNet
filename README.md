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
  4. (512, 256) image 
     - process time ~50ms (~20fps)
     

BJNet_3rd
  1. fused cost volume part & cascade volume part & hourglass 제거
  2. concat cost volume generation 제거
  
  
BJNet_4th
  1. gwc cost volume modified<br>
     torch.reciprocal(fea1 - mea2 + 1e-18) -> not good   

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
