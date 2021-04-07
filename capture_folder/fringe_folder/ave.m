function pjj=ave(imgdir,imgnum,pixelsize,gammaflag,deta,winsize,Imgref)
%%if gammaflag==1 do gamma filter
imgDir=imgdir;  
oldPwd = pwd;
pjj=zeros(pixelsize(1),pixelsize(2),imgnum);

for i=1:imgnum                                           %% Æ½¾ùÔÆÎÆÍ¼Ïñ
cd([imgDir,num2str(i),'\']);   
x = dir;  
listOfImages = [];  
  for j = 1:length(x),  
     if x(j).isdir == 0,  
       listOfImages = [listOfImages; x(j)];  
     end;  
  end;  
  
  for j = 1:length(listOfImages)  
    fileName = listOfImages(j).name;  
    rfid=[imgDir,num2str(i),'\',fileName];  
    Irgb=imread(rfid);  
    Iset{j}=Irgb;  
  end 

  pj=zeros(size(Iset{1}));
  for j = 1:length(listOfImages)  
       pj=pj+double(Iset{j});
  end
  pj=pj/length(listOfImages);
  if gammaflag==1
      pj=gamma_filter(pj,Imgref,deta,winsize);
  end
  pjj(:,:,i)=pj;
end       

cd(oldPwd);
end
