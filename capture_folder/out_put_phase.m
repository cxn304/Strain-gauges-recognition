
%%
%参数设置
imgDir='./moire_img/';    %总文件夹
files = dir([imgDir,'*.','png']);
len = length(files);
lfile = {};
rfile = {};
for i = 1:len  % 区分左相机和右相机拍摄的图片
    if strfind(files(i).name , 'l')
        lfile{i} = files(i).name;
    else
        rfile{i-len/2} = files(i).name;
    end
end
% obj=input('待测物子文件夹名','s');
% calibsub=input('标定文件夹名','s');
% name={'L\heng\','L\shu\','R\heng\','R\shu\'};
% namewrap={'L_heng','L_shu','R_heng','R_shu'};
% imgnum=4;                         %相移张数
% gammaflag=0;%1=on;
% deta=[];
% winsize=[];
% if isempty(gammaflag)||gammaflag==0
%     gammaflag=0;
% else
%     gammaflag=1;
%     deta=input('输入sigma滤波参数:');
%     if isempty(deta)
%         deta=3;
%     end
%     winsize=input('输入sigma滤波窗口:');
%     if isempty(deta)
%         winsize=7;
%     end
% end


lshizi=imread([imgDir,lfile{9}]); %  左图十字
lwu = imread([imgDir,lfile{10}]); %  左图无十字
[clx,cly,pixelsize] = find_cross_point(lshizi,lwu);
rshizi=imread([imgDir,rfile{9}]);
rwu = imread([imgDir,rfile{10}]);
[crx,cry,~] = find_cross_point(rshizi,rwu);
%%
%初始化参数
% figure(1)
% zpl=zerophase([imgDir,lfile{9}]);  %左图十字坐标
% figure(2)
% zpr=zerophase([imgDir,rfile{9}]);  %右图十字座标

cutflag=0;
calibcutflag=0;
if (strcmp(input('按任意键进行图片剪裁(左上到右下);回车取消','s'),'') )
    hlc=[1 pixelsize(1)];
    slc=[1 pixelsize(2)];
    src=slc;hrc=hlc;
    height=hlc(2)-hlc(1)+1;
    width=slc(2)-slc(1)+1;
else
    hlc=clx;
    slc=cly;
    hrc = crx;
    src = cry;
    
    height=max(hlc(2)-hlc(1),hrc(2)-hrc(1))+1;%使左右图大小相等??什么意思
    width=max(slc(2)-slc(1),src(2)-src(1))+1;
    hlc(2)=hlc(1)+height-1;
    hrc(2)=hrc(1)+height-1;
    slc(2)=slc(1)+width-1;
    src(2)=src(1)+width-1;
    if (strcmp(input('按回车键剪裁标定图片','s'),'') )
        calibcutflag=1;
        %calib cut
    end
end

maskflag=0;
if (strcmp(input('按回车键创建掩膜','s'),'') )
    maskflag=1;
    sl=imread([imgDir,lfile{9}]);
    slmask=double(imread([imgDir,obj,'\slmask.bmp']));
    maskl=masked(sl(hlc(1):hlc(2),slc(1):slc(2)));
    maskl(slmask==0)=nan;
    srmask=double(imread([imgDir,obj,'\srmask.bmp']));
    sr=imread([imgDir,rfile{9}]);
    maskr=masked(sr(hrc(1):hrc(2),src(1):src(2)));
    srmask=double(imread([imgDir,obj,'\srmask.bmp']));
    maskr(srmask==0)=nan;
end

zpl=zpl+1-[hlc(1),slc(1)];
zpr=zpr+1-[hrc(1),src(1)];

if calibcutflag==1
    calibcut(imgDir,calibsub);
end
zhouqi=zeros(1,4);
for i=1:4  %name
    avepics=ave([imgDir,obj,'\',name{i}],imgnum,pixelsize,gammaflag,deta,winsize,[]);%求平均
    if i<3
        avepics=avepics(hlc(1):hlc(2),slc(1):slc(2),:);
        zp=zpl;
        sss=0;
        if maskflag==1
            im_mask=maskl;
        end
    else
        avepics=avepics(hrc(1):hrc(2),src(1):src(2),:);
        zp=zpr;
        sss=1;
        if maskflag==1
            im_mask=maskr;
        end
    end
    [phi,im_mag]=fourstepbasedphase(avepics,imgnum);
    clear avepics
    % save([imgDir,obj,'\',namewrap{i},'.asc'],'phi','-ASCII');
    deri=derical(phi,im_mask,zp);
    figure(10)
    imagesc(deri)
    unwrap=goodscan(deri,im_mask,phi,sss);
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
    clear IM im_mag
    zhouqi(i)=unwrap(zp(1),zp(2))/pi;
    unwrap=unwrap-round(zhouqi(i))*pi;%横竖有关
    
    eval([namewrap{i} ' = unwrap;']);
    save([imgDir,obj,'\',namewrap{i},'.mat'],namewrap{i});
    eval(['clear ',namewrap{i},';']);
    figure(i)
    imagesc(unwrap);
    clear unwrap
end
%%
function [xx,yy,pixelsize] = find_cross_point(lshizi,lwu)
% 自动识别十字叉中心点坐标
zuoshizitu = lwu-lshizi;
pixelsize=size(lshizi);    %初始图像尺寸1024,1280
zuoshizitu = imbinarize(zuoshizitu);
[H,I,R]=hough(zuoshizitu);
Peaks=houghpeaks(H,2);  % 就找两条最明显的直线
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
% 四个点的两个直线求交点
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
%8像素pixelstep=8
%16像素pixelstep=16 每四相移之间的周期（2pi）间隔

row=size(aveimg,1);
colomn=size(aveimg,2);
cycle=fix(pixelstep/4);
% 计算相位

fi1=zeros(row,colomn);
fi2=zeros(row,colomn);
fi1=atan2(aveimg(:,:,4)-aveimg(:,:,2),aveimg(:,:,1)-aveimg(:,:,3));%atan2(y,x)=atan(y/x),返回-pi到pi之间的值  (-pi,pi]
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
function IM_unwrapped = GuidedFloodFill3(IM_phase, IM_unwrapped, adjoin ,derivative_variance)

if size(adjoin,1)==0
    IM_unwrapped =IM_unwrapped ;
else
    k=0;%adjoin指针%%!!!!!!!!!!!!!!!!!!!!!!!!!!!对棋盘格不适用，需要扩展四个角
    while ~isnan(adjoin(k+1,1))
        k=k+1;
    end
    %%derivative_varianc,IM_phase,IM_unwrapped are masked
    [r_dim, c_dim] = size(IM_phase);
    if ~isempty(derivative_variance)
        % Include edge pixels%未验证
        adjoinmap=ones([r_dim, c_dim]);
        adjoinmap(adjoin(1:k))=0;%若矩阵adjoin中为0，则已经在adjoin list中
        
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
                    
                    phase_ref(1) = IM_unwrapped(r_active+1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference unwrapped phase
                    qualityneighbor(1)=derivative_variance(r_active+1, c_active);
                    
                else % unwrapped_binary(r_active+1, c_active)==0未展开
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
                    %Obtain the reference unwrapped phase
                    %       D = IM_phase(r_active, c_active)-phase_ref;
                    %       deltap = atan2(sin(D),cos(D));   % Make it modulo +/-pi
                    %       phasev(3) = phase_ref + deltap;  % This is the unwrapped phase
                    %       IM_magv(3)= IM_mag(r_active, c_active+1);
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
                    %Obtain the reference unwrapped phase
                    %       D = IM_phase(r_active, c_active)-phase_ref;
                    %       deltap = atan2(sin(D),cos(D));   % Make it modulo +/-pi
                    %       phasev(4) = phase_ref + deltap;  % This is the unwrapped phase
                    %       IM_magv(4)= IM_mag(r_active, c_active-1);
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
            
            %     if isnan(phase_ref(m))
            %         k
            %         qualityneighbor
            %         phase_ref
            %         r_active
            %         c_active
            %     end
            
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
                else % unwrapped_binary(r_active+1, c_active)==0未展开
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
            
            
            %end
            
        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
        
    end
    return
end
end
%%
%phase derivative map
function [A,Aorigin]=derical(phi,im_mask,zp)
%phi阶段相位，im_mask掩膜；zp十字中心
%质量图，【0,1】，1比较好；后面的1.5；0.9可调
dimx=size(phi,1);
dimy=size(phi,2);
A=zeros(size(phi,1),size(phi,2));
du=phi(3:size(phi,1),2:size(phi,2)-1)-phi(2:size(phi,1)-1,2:size(phi,2)-1);
du=du-2*pi*round(du/2/pi);

mean_du = (du(2:dimx-3, 2:dimy-3)+du(2:dimx-3,1:dimy-4)+du(...
    2:dimx-3,3:dimy-2)+du(1:dimx-4,2:dimy-3)+du(3:dimx-2,2:dimy-3))./5;
% stdvaru = sqrt( (du(3:dimx-2, 3:dimy-2) - mean_du).^2 + (du(3:dimx-2,2:dimy-3) - mean_du).^2 + ...
%               (du(3:dimx-2,4:dimy-1) - mean_du).^2 + (du(2:dimx-3,3:dimy-2) - mean_du).^2 + (du(4:dimx-1,3:dimy-2) - mean_du).^2 );

dv=phi(2:size(phi,1)-1,3:size(phi,2))-phi(2:size(phi,1)-1,2:size(...
    phi,2)-1);
dv=dv-2*pi*round(dv/2/pi);

mean_dv = (du(2:dimx-3, 2:dimy-3)+ du(2:dimx-3,1:dimy-4)+ du(...
    2:dimx-3,3:dimy-2)+ du(1:dimx-4,2:dimy-3)+du(3:dimx-2,2:dimy-3))./5;
% stdvarv = sqrt( (dv(3:dimx-2, 3:dimy-2) - mean_dv).^2 + (dv(3:dimx-2,2:dimy-3) - mean_dv).^2 + ...
%               (dv(3:dimx-2,4:dimy-1) - mean_dv).^2 + (dv(2:dimx-3,3:dimy-2) - mean_dv).^2 + (dv(4:dimx-1,3:dimy-2) - mean_dv).^2 );

A(3:dimx-2, 3:dimy-2)=(sqrt( (du(2:dimx-3, 2:dimy-3) - mean_du).^2 + (...
    du(2:dimx-3,1:dimy-4) - mean_du).^2 + ...
    (du(2:dimx-3,3:dimy-2) - mean_du).^2 + (du(1:dimx-4,2:dimy-3) - mean_du).^2 + (du(3:dimx-2,2:dimy-3) - mean_du).^2 )+...
    sqrt( (dv(2:dimx-3, 2:dimy-3) - mean_dv).^2 + (dv(2:dimx-3,1:dimy-4) - mean_dv).^2 + ...
    (dv(2:dimx-3,3:dimy-2) - mean_dv).^2 + (dv(1:dimx-4,2:dimy-3) - mean_dv).^2 + (dv(3:dimx-2,2:dimy-3) - mean_dv).^2 ))/4;

% A(3:dimx-2, 3:dimy-2)=0.25./max(A(3:dimx-2, 3:dimy-2),0.25);
Aorigin=A;%噪声大小没有上限，噪声越大A越大；
yuzhiqu=A(zp(1)-5:zp(1)+5,zp(2)-5:zp(2)+5);
yuzhiqu=sort(yuzhiqu(:));
yuzhi=yuzhiqu(fix(sum(~isnan(yuzhiqu))*0.9))*1.5;

A=yuzhi./max(A,yuzhi);
return;
end
%%
function A=masked(img)
%通过亮度对比，创建简单的掩膜
%A:计算区域=1；非计算区域=nan；
%调整结构元素
    A=imbinarize(img);
    A=imopen(A,strel('square',3));
    A=imclose(A,strel('square',2));
    A=imopen(A,strel('square',9));
    imshow(A)%显示该图
    set(gcf,'outerposition',get(0,'screensize'));
end
