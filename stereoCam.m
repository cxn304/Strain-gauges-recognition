close all;
toopencv(stereoParams);
% imageDir = 'D:\cxn_project\Strain-gauges-recognition\cali_img';
% leftImages = imageDatastore(fullfile(imageDir,'left'));
% rightImages = imageDatastore(fullfile(imageDir,'right'));

%%
% Detect the checkerboards.
% [imagePoints,boardSize] = detectCheckerboardPoints(...
%     leftImages.Files,rightImages.Files);
%%
% Specify world coordinates of checkerboard keypoints.
% squareSizeInMillimeters = 10;
% worldPoints = generateCheckerboardPoints(boardSize,squareSizeInMillimeters);
%%
% Read in the images.
% I1 = readimage(leftImages,26);
% I2 = readimage(rightImages,26);
I1 = imread('D:\cxn_project\Strain-gauges-recognition\cali_img\left\l18.bmp');
I2 = imread('D:\cxn_project\Strain-gauges-recognition\cali_img\right\r18.bmp');
imageSize = [size(I1,1),size(I1,2)];
%%
% Calibrate the stereo camera system.
% stereoParams = estimateCameraParameters(imagePoints,worldPoints, ...
%                                         'ImageSize',imageSize);

%%
% Rectify the images using 'full' output view.
[J1_full,J2_full] = rectifyStereoImages(I1,I2,stereoParams, ...
  'cubic','OutputView','full');
%%
% Display the result for 'full' output view.
figure; 
imshow(stereoAnaglyph(J1_full,J2_full));
%%
% Rectify the images using 'valid' output view. This is most suitable
% for computing disparity.
[J1_valid,J2_valid] = rectifyStereoImages(I1,I2,stereoParams, ...
  'cubic','OutputView','valid');
%%
% Display the result for 'valid' output view.
figure; 
imshow(stereoAnaglyph(J1_valid,J2_valid));

% toopencv(stereoParams);
function []= toopencv(stereoParams)
    RD1 = stereoParams.CameraParameters1.RadialDistortion;
    TD1 = stereoParams.CameraParameters1.TangentialDistortion;
    D1 = [RD1(1), RD1(2), TD1(1), TD1(2), RD1(3)];
    K1 = stereoParams.CameraParameters1.IntrinsicMatrix';

    RD2 = stereoParams.CameraParameters2.RadialDistortion;
    TD2 = stereoParams.CameraParameters2.TangentialDistortion;
    D2 = [RD2(1), RD2(2), TD2(1), TD2(2), RD2(3)];
    K2 = stereoParams.CameraParameters2.IntrinsicMatrix';

    size = stereoParams.CameraParameters1.ImageSize;

    rot = stereoParams.RotationOfCamera2;
    trans = stereoParams.TranslationOfCamera2;
    %ИЎДж
%     T=eye(4);
%     T(1:3,1:3)=rot;
%     T(1:3,4)=trans;
%     T=inv(T);
%     rot=T(1:3,1:3);
%     trans=T(1:3,4);

    stereoParameters = struct('K1',K1 ,'D1', D1,'K2',K2 ,...
        'D2', D2,'size',size, 'rot',rot,'trans',trans);
    save('./internal_reference/stereoParameters.mat','stereoParameters');
end