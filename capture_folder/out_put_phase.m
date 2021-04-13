clear
%%
%��������
imgDir='./moire_img/';    %���ļ���
usefolders = find_folders(imgDir);
len = length(usefolders);
for iii = 1:len
    nowdir = [imgDir usefolders{iii} '/'];
    files = dir([nowdir,'*.','png']);
    imglen = length(files);
    lfile = {};
    rfile = {};
    allfile = {};
    for i = 1:imglen  % ���������������������ͼƬ
        if strfind(files(i).name , 'l')
            lfile{i} = files(i).name;
        else
            rfile{i-imglen/2} = files(i).name;
        end
        allfile{i} = files(i).name;
    end
    
    lshizi=imread([nowdir,lfile{9}]); %  ��ͼʮ��
    lwu = imread([nowdir,lfile{10}]); %  ��ͼ��ʮ��
    %     [clx,cly,pixelsize] = find_cross_point(lshizi,uint8(mean(avepics,3)));
    imshow(lshizi)
    [clx,cly] = ginput(1);
    clx = int32(clx);
    cly = int32(cly);
    rshizi=imread([nowdir,rfile{9}]);
    rwu = imread([nowdir,rfile{10}]);
    %     [crx,cry,~] = find_cross_point(rshizi,rwu);
    imshow(rshizi)
    [crx,cry] = ginput(1);
    crx = int32(crx);
    cry = int32(cry);
    [height,width] = size(lshizi);
    [maskl,maskr] = find_intersection_area(clx,cly,crx,cry,height,width);
    
    zhouqi=zeros(1,4);
    avepics = zeros(height,width,4);
    for i=1:4  % ѭ������{'L\heng\','L\shu\','R\heng\','R\shu\'}
        [zp,im_mask,sss,avepics] = create_avgimg(nowdir,allfile,i,avepics...
            ,maskl,clx,cly,crx,cry);
        [phi,im_mag]=fourstepbasedphase(avepics,4); % phi��ԭͼά��һ��
        thing_mask = find_specie(phi,height,width);
        % phi�ǽ��������λ
        clear avepics
        % save([imgDir,obj,'\',namewrap{i},'.asc'],'phi','-ASCII');
        [deri,~]=derical(phi,thing_mask,zp);
        figure
        imagesc(deri)
        % imagesc(A) ������A�е�Ԫ����ֵ����Сת��Ϊ��ͬ��ɫ��
        % �����������Ӧλ�ô���������ɫȾɫ
        unwrap=goodscan(deri,thing_mask,phi,sss,width);
        A=~isnan(unwrap);
        % A=~isnan((0./fix(deri)+1).*mask);
        se=strel('diamond',1);
        A=(imdilate(A,se)-A).*im_mask;
        l=find(A(:)==1);
        adjoin=nan(length(l)*5,1);
        adjoin(1:length(l),1)=l;
        clear l
        phi=im_mask.*phi;
        unwrap=GuidedFloodFill3(phi, unwrap, adjoin ,deri);
        zhouqi(i)=unwrap(zp(1),zp(2))/pi;
        unwrap=unwrap-round(zhouqi(i))*pi;%�����й�
        figure(i)
        imagesc(unwrap);
        clear unwrap
    end
end
%%
function img = find_specie(phi,height,width)
% �ҳ�����λ��
b=ones(height,width);
b(phi==0)=0;
se1=strel('disk',2);
b=imopen(b,se1);
b=imclose(b,se1);
imLabel = bwlabel(b);                %�Ը���ͨ����б��
stats = regionprops(imLabel,'Area');    %�����ͨ��Ĵ�С
area = cat(1,stats.Area);
index = find(area == max(area));        %�������ͨ�������
img = ismember(imLabel,index);          %��ȡ�����ͨ��ͼ��
end
%%
function [usefolders] = find_folders(dirs)
% ���folder
folders = dir(dirs);
usefolders = {};
k=1;
for i = 1:length(folders)
    if ~isempty(strfind(folders(i).name,'16'))
        usefolders{k}=folders(i).name;
        k = k+1;
    end
end
end
%%
function [zp,im_mask,sss,avepics] = create_avgimg(nowdir,allfile,i,avepics...
    ,maskl,clx,cly,crx,cry)
% ����ƽ��ͼ�񲢷���ʮ����������ͼ���������Ĥ
if i<=2  % ǰ2������
    avepics(:,:,1) = imread([nowdir,allfile{4*(i-1)+1}]);
    avepics(:,:,2) = imread([nowdir,allfile{4*(i-1)+2}]);
    avepics(:,:,3) = imread([nowdir,allfile{4*(i-1)+3}]);
    avepics(:,:,4) = imread([nowdir,allfile{4*(i-1)+4}]);
    avepics=avepics.*maskl;
    zp=[clx,cly]; % zp��ʾʮ������
    sss=0;
    im_mask=maskl;
else   % ���2������
    avepics(:,:,1) = imread([nowdir,allfile{4*(i-1)+1+2}]);
    avepics(:,:,2) = imread([nowdir,allfile{4*(i-1)+2+2}]);
    avepics(:,:,3) = imread([nowdir,allfile{4*(i-1)+3+2}]);
    avepics(:,:,4) = imread([nowdir,allfile{4*(i-1)+4+2}]);
    avepics=avepics.*maskr;
    zp=[crx,cry];
    sss=1;
    im_mask=maskr;
end
end
%%
function [xx,yy,pixelsize] = find_cross_point(lshizi,lwu)
% �Զ�ʶ��ʮ�ֲ����ĵ�����
lwu(find(lwu>10)) = lwu(find(lwu>10))+7;    % �д�����
zuoshizitu = lwu-lshizi;
pixelsize=size(lshizi);    %��ʼͼ��ߴ�1024,1280
zuoshizitu = imbinarize(zuoshizitu);
[H,I,R]=hough(zuoshizitu);
Peaks=houghpeaks(H,2);  % �������������Ե�ֱ��
lines=houghlines(zuoshizitu,I,R,Peaks);
figure
imshow(lwu-lshizi)
hold on;
for k=1:length(lines)
    xy=[lines(k).point1;lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',1);
end
[xx,yy] = cal_node(lines(1).point1,lines(1).point2,...
    lines(2).point1,lines(2).point2);
xx = round(xx);
yy = round(yy);
plot(xx,yy,'r*')
end
%%
function [X,Y]= cal_node( X1,Y1,X2,Y2 )
% �ĸ��������ֱ���󽻵�
if X1(1)==Y1(1)
    X=X1(1);
    k2=(Y2(2)-X2(2))/(Y2(1)-X2(1));
    b2=X2(2)-k2*X2(1);
    Y=k2*X+b2;
end
if X2(1)==Y2(1)
    X=X2(1);
    k1=(Y1(2)-X1(2))/(Y1(1)-X1(1));
    b1=X1(2)-k1*X1(1);
    Y=k1*X+b1;
end
if X1(1)~=Y1(1)&&X2(1)~=Y2(1)
    k1=(Y1(2)-X1(2))/(Y1(1)-X1(1));
    k2=(Y2(2)-X2(2))/(Y2(1)-X2(1));
    b1=X1(2)-k1*X1(1);
    b2=X2(2)-k2*X2(1);
    if k1==k2
        X=[];
        Y=[];
    else
        X=(b2-b1)/(k1-k2);
        Y=k1*X+b1;
    end
end
end
%%
function [IM,im_mag]=fourstepbasedphase(aveimg,pixelstep)
%8����pixelstep=8
%16����pixelstep=16 ÿ������֮������ڣ�2pi�����
cycle=fix(pixelstep/4);
% ������λ
fi1=atan2(aveimg(:,:,4)-aveimg(:,:,2),aveimg(:,:,1)-aveimg(:,:,3));%atan2(y,x)=atan(y/x),����-pi��pi֮���ֵ  (-pi,pi]
im_mag=(aveimg(:,:,4)-aveimg(:,:,2)).^2+(aveimg(:,:,1)-aveimg(:,:,3)).^2;
IM=fi1;
for k=2:cycle
    fi2=atan2(aveimg(:,:,4*k)-aveimg(:,:,4*k-2),aveimg(:,:,4*k-3)-aveimg(:,:,4*k-1))-2*pi/pixelstep*(k-1);
    fi2=-ceil((fi2-pi)/pi/2)*2*pi+fi2;
    fi2=fi2+2*pi*fix((fi1-fi2)/pi);
    IM=IM+fi2;
end
IM=IM/cycle;
end
%%
function IM_unwrapped = GuidedFloodFill3(IM_phase, IM_unwrapped, ...
    adjoin ,derivative_variance)

if size(adjoin,1)==0
    IM_unwrapped =IM_unwrapped ;
else
    k=0;%adjoinָ��%%!!!!!!!!!!!!!!!!!!!!!!!!!!!�����̸����ã���Ҫ��չ�ĸ���
    while ~isnan(adjoin(k+1,1))
        k=k+1;
    end
    %%derivative_varianc,IM_phase,IM_unwrapped are masked
    [r_dim, c_dim] = size(IM_phase);
    if ~isempty(derivative_variance)
        % Include edge pixels%δ��֤
        adjoinmap=ones([r_dim, c_dim]);
        adjoinmap(adjoin(1:k))=0;%������adjoin��Ϊ0�����Ѿ���adjoin list��
        
        while k~= 0
            %Loop until there are no more adjoining pixels %Derivative variance values of the adjoining pixels (pad the zero adjoining values with 100)
            [max_deriv_var,ma] = max(derivative_variance(adjoin(k:-1:1))); % the minimum derivative variance
            [r_active, c_active]=ind2sub(size(derivative_variance),adjoin(k+1-ma));
            adjoin(k+1-ma)=adjoin(k);
            adjoin(k)=NaN;
            k=k-1;
            
            phase_ref = nan(1,4);     % Initialize.  Will overwrite for valid pixels
            qualityneighbor= nan(1,4);
            %IM_magv  = nan(1,4);     % Initialize.  Will overwrite for valid pixels
            %First search below for an adjoining unwrapped phase pixel
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active+1, c_active))
                    
                    phase_ref(1) = IM_unwrapped(r_active+1, c_active)...
                        +IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference unwrapped phase
                    qualityneighbor(1)=derivative_variance(r_active+1, c_active);
                    
                else % unwrapped_binary(r_active+1, c_active)==0δչ��
                    if(~isnan(IM_phase(r_active+1, c_active))*adjoinmap(r_active+1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active+1, c_active);  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        adjoinmap(r_active+1, c_active)=0;
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active-1, c_active))==1
                    
                    phase_ref(2) = IM_unwrapped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                    qualityneighbor(2)=derivative_variance(r_active-1, c_active);
                    %Obtain the reference unwrapped phase
                    %       D = IM_phase(r_active, c_active)-phase_ref;
                    %       deltap = atan2(sin(D),cos(D));   % Make it modulo +/-pi
                    %       phasev(2) = phase_ref + deltap;  % This is the unwrapped phase
                    %       IM_magv(2)= IM_mag(r_active-1, c_active);
                else % unwrapped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active))*adjoinmap(r_active-1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active-1, c_active);
                        adjoinmap(r_active-1, c_active)=0;
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active+1))
                    
                    phase_ref(3) = IM_unwrapped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                    qualityneighbor(3)=derivative_variance(r_active, c_active+1);

                else % unwrapped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1))*adjoinmap(r_active, c_active+1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active+1);
                        adjoinmap(r_active, c_active+1)=0;
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active-1))
                    
                    phase_ref(4) = IM_unwrapped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                    qualityneighbor(4)=derivative_variance(r_active, c_active-1);

                else % unwrapped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1))*adjoinmap(r_active, c_active-1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active-1);
                        adjoinmap(r_active, c_active-1)=0;
                    end
                end
            end
            
            [IM_max,m] = max(qualityneighbor);
            %     idx_max = find((IM_magv >= 0.99*IM_max) & (idx_del==1));
            IM_unwrapped(r_active, c_active) =phase_ref(m);  % Use the first, if there is a tie

        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    else
        
        while k~= 0
            %input adjoin should be a pixel position i.e [3,3]
            %before run, add unwrapped start point(rowref, colref)and adjoin points
            %   if im_mask(rowref-1, colref, 1)==1;  adjoin=[rowref-1, colref;adjoin] end       %Mark the pixels adjoining the selected point
            % if im_mask(rowref+1, colref, 1)==1;  adjoin(rowref+1, colref, 1) = 1; end
            % if im_mask(rowref, colref-1, 1)==1;  adjoin(rowref, colref-1, 1) = 1; end
            % if im_mask(rowref, colref+1, 1)==1;  adjoin(rowref, colref+1, 1) = 1; end
            r_active=adjoin(k,1);
            c_active=adjoin(k,2);
            adjoin(k,:)=NaN;
            k=k-1;
            
            
            
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active+1, c_active))
                    phase_ref = IM_unwrapped(r_active+1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference unwrapped phase
                else % unwrapped_binary(r_active+1, c_active)==0δչ��
                    if(~isnan(IM_phase(r_active+1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active+1, c_active];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active-1, c_active))
                    phase_ref= IM_unwrapped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                else % unwrapped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active-1, c_active];
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active+1))
                    phase_ref = IM_unwrapped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                else % unwrapped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active+1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active-1))
                    phase_ref= IM_unwrapped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                else % unwrapped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active-1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end

            IM_unwrapped(r_active, c_active) =phase_ref;

        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    end
    return
end
end
%%
%phase derivative map
function [A,Aorigin]=derical(phi,im_mask,zp)
%phi�׶���λ��im_mask��Ĥ��zpʮ������
%����ͼ����0,1����1�ȽϺã������1.5��0.9�ɵ�
row=size(phi,1);   % row
col=size(phi,2);   % col
A=zeros(row,col);
du=phi(3:size(phi,1),2:size(phi,2)-1)-phi(2:size(phi,1)-1,2:size(phi,2)-1);
du=du-2*pi*round(du/2/pi);

mean_du = (du(2:row-3, 2:col-3)+du(2:row-3,1:col-4)+du(...
    2:row-3,3:col-2)+du(1:row-4,2:col-3)+du(3:row-2,2:col-3))./5;
% stdvaru = sqrt( (du(3:row-2, 3:col-2) - mean_du).^2 + (du(3:row-2,2:col-3) - mean_du).^2 + ...
%               (du(3:row-2,4:col-1) - mean_du).^2 + (du(2:row-3,3:col-2) - mean_du).^2 + (du(4:row-1,3:col-2) - mean_du).^2 );

dv=phi(2:size(phi,1)-1,3:size(phi,2))-phi(2:size(phi,1)-1,2:size(...
    phi,2)-1);
dv=dv-2*pi*round(dv/2/pi);

mean_dv = (du(2:row-3, 2:col-3)+ du(2:row-3,1:col-4)+ du(...
    2:row-3,3:col-2)+ du(1:row-4,2:col-3)+du(3:row-2,2:col-3))./5;
% stdvarv = sqrt( (dv(3:row-2, 3:col-2) - mean_dv).^2 + (dv(3:row-2,2:col-3) - mean_dv).^2 + ...
%               (dv(3:row-2,4:col-1) - mean_dv).^2 + (dv(2:row-3,3:col-2) - mean_dv).^2 + (dv(4:row-1,3:col-2) - mean_dv).^2 );

A(3:row-2, 3:col-2)=(sqrt( (du(2:row-3, 2:col-3) - mean_du).^2 + (...
    du(2:row-3,1:col-4) - mean_du).^2 + ...
    (du(2:row-3,3:col-2) - mean_du).^2 + (du(1:row-4,2:col-3) -...
    mean_du).^2 + (du(3:row-2,2:col-3) - mean_du).^2 )+...
    sqrt( (dv(2:row-3, 2:col-3) - mean_dv).^2 + (dv(2:row-3,1:col-4)...
    - mean_dv).^2 + ...
    (dv(2:row-3,3:col-2) - mean_dv).^2 + (dv(1:row-4,2:col-3)...
    - mean_dv).^2 + (dv(3:row-2,2:col-3) - mean_dv).^2 ))/4;

% A(3:row-2, 3:col-2)=0.25./max(A(3:row-2, 3:col-2),0.25);
Aorigin=A;%������Сû�����ޣ�����Խ��AԽ��
yuzhiqu=A(zp(1)-5:zp(1)+5,zp(2)-5:zp(2)+5);
yuzhiqu=sort(yuzhiqu(:));
yuzhi=yuzhiqu(fix(sum(~isnan(yuzhiqu))*0.9))*1.5;

A=yuzhi./max(A,yuzhi);
end
%%
function A=masked(img)
%ͨ�����ȶԱȣ������򵥵���Ĥ
%A:��������=1���Ǽ�������=nan��
%�����ṹԪ��
A=imbinarize(img);
A=imopen(A,strel('square',3));
A=imclose(A,strel('square',2));
A=imopen(A,strel('square',9));
imshow(A)%��ʾ��ͼ
set(gcf,'outerposition',get(0,'screensize'));
end
%% �ҳ����������������
function [maskl,maskr]=find_intersection_area(clx,cly,crx,cry,height,width)
if clx>crx
    maxwl = width;
    minwl = clx - crx;
    maxwr = width - (clx-crx);
    minwr = 1;
else
    maxwl = width - (crx-clx);
    minwl = 1;
    maxwr = width;
    minwr = crx-clx;
end
if cly>cry
    maxhl = height;
    minhl = cly - cry;
    maxhr = height - (cly - cry);
    minhr = 1;
else
    maxhl = height - (cry-cly);
    minhl = 1;
    maxhr = height;
    minhr = cry-cly;
end
maskl = zeros(height,width);
maskl(minhl:maxhl,minwl:maxwl) = 1;
maskr = zeros(height,width);
maskr(minhr:maxhr,minwr:maxwr) = 1;
end
%%
function pjj=ave(imgdir,imgnum,pixelsize,gammaflag,deta,winsize,Imgref)
%%if gammaflag==1 do gamma filter��deta��winsize,Imgref���ù�
%imgdir:·����imgnum�����ƴ�����pixelsize��ͼƬ��С����ά�����У���
%pjj:ƽ�������ά����ͼƬ��*ͼƬ��*��������
imgDir=imgdir;
oldPwd = pwd;
pjj=zeros(pixelsize(1),pixelsize(2),imgnum);

for i=1:imgnum                                           %% ƽ������ͼ��
    cd([imgDir,num2str(i),'\']);
    x = dir;
    listOfImages = [];
    for j = 1:length(x)
        if x(j).isdir == 0
            listOfImages = [listOfImages; x(j)];
        end
    end
    
    for j = 1:length(listOfImages)
        fileName = listOfImages(j).name;
        rfid=[imgDir,num2str(i),'\',fileName];
        Irgb=imread(rfid);
        Iset{j}=Irgb;
    end
    
    pj=zeros(size(Iset{1}));
    for j = 1:length(listOfImages)
        ISE=double(Iset{j});
        pj=pj+ISE;
    end
    pj=pj/length(listOfImages);   % pj��ʾһ������������ͼ���ƽ��
    pj(pj>254)=nan;
    if gammaflag==1
        pj=gamma_filter(pj,Imgref,deta,winsize);
    end
    pjj(:,:,i)=pj;
end

cd(oldPwd);
end
%%
function unwrap=goodscan(deri,mask,phi,sss,width)
% maxline=20;
dimx=size(phi,1);   % row
dimy=size(phi,2);   % col
% linumber=dimx*3;
mask=(0./fix(deri)+1).*mask;
% [r,rowref ]=max(sum(mask,2));
phi=mask.*phi;%phi; area of quality==1
C=~isnan(phi);%�ԷǼ�����������Ͳ������ҳ���ͨ·��A4=imdilate(A3,se)
cloumnref = width-5;
cc=C(:,cloumnref);
k=find(cc==1);
rowref=k(fix(numel(k)/2));
colref=cloumnref;
edl=cloumnref+2;
unwrap=nan(dimx,dimy);
unwrap(rowref,colref)=phi(rowref, colref);
adjoin=nan(dimx*(edl-cloumnref+1),2);
k=0;
if ~isnan(phi(rowref, colref+1))==1&&colref+1<=edl
    k=k+1;adjoin(k,:)=[rowref, 2];
end
if ~isnan(phi(rowref-1, colref))==1
    k=k+1;adjoin(k,:)=[rowref-1, 1];
end       %Mark the pixels adjoining the selected point
if ~isnan(phi(rowref+1, colref))==1
    k=k+1;adjoin(k,:)=[rowref+1, 1];
end
unwrap(:,cloumnref:edl) = GuidedFloodFill3(phi(:,cloumnref:edl), ...
    unwrap(:,cloumnref:edl), adjoin,[]);
adjoin=nan(dimx,1);
if edl==dimy-2
else
    for i=edl+1:dimy-2
        unwrap(:,i)=unwrap(:,i-1)+phi(:,i)-phi(:,i-1)-2*pi*round(...
            (phi(:,i)-phi(:,i-1))/2/pi);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))...
            ==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),1);%up
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==...
            -1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),-1);
    end
end
if cloumnref==3
else
    for i=cloumnref-1:-1:3
        unwrap(:,i)=unwrap(:,i+1)+phi(:,i)-phi(:,i+1)-2*pi*round((phi(:,i)-phi(:,i+1))/2/pi);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),1);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==-1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),-1);
    end
end
end
%%
function IM_unwrapped =unwrapline(IM_phase,IM_unwrapped,adjoin,k,direction)
r_dim= length(IM_phase);
%%%%%%%%%adjoin���Ǹ������Ƿ����
if direction==-1%downwards
    while k~=0 %Loop until there are no more adjoining pixels
        r_active=adjoin(k);
        adjoin(k)=NaN;
        k=k-1;  
        %Then search above
        if(r_active-1>=1)  % Is this a valid index?
            if ~isnan(IM_unwrapped(r_active-1))
                IM_unwrapped(r_active)= IM_unwrapped(r_active-1)...
                    +IM_phase(r_active)-IM_phase(r_active-1)...
                    -2*pi*round((IM_phase(r_active)-IM_phase(...
                    r_active-1))/2/pi);
                if (r_active+1<=r_dim)
                    if(~isnan(IM_phase(r_active+1)))
                        k=k+1;
                        adjoin(k)=r_active+1; % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
        end
        
    end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
elseif direction==1%'upwards'
    while k~=0 %Loop until there are no more adjoining pixels
        r_active=adjoin(k);
        adjoin(k)=NaN;
        k=k-1;
        
        if(r_active+1<=r_dim)  % Is this a valid index?
            if ~isnan(IM_unwrapped(r_active+1))
                IM_unwrapped(r_active) = IM_unwrapped(r_active+1)...
                    +IM_phase(r_active)-IM_phase(r_active+1)...
                    -2*pi*round((IM_phase(r_active)-IM_phase(...
                    r_active+1))/2/pi);       % Obtain the reference unwrapped phase
                if (r_active-1>0)
                    if(~isnan(IM_phase(r_active-1)))
                        k=k+1;
                        adjoin(k)=r_active-1;
                    end
                end
            end
        end
    end
end
return
end
