clear %得到相位图后匹配点像素坐标,三维重建中间步骤
path='./calculate_dis/qn/';%相位图地址

%%

load([path 'unwrap2.mat']);
% unwrap=unwrap+(0.2/255)*randn(size(unwrap));
L_shu = unwrap;
load([path 'unwrap4.mat']);
% unwrap=unwrap+(0.2/255)*randn(size(unwrap));
R_shu = unwrap;
load([path 'unwrap1.mat']);
% unwrap=unwrap+(0.2/255)*randn(size(unwrap));
L_heng = unwrap;
load([path 'unwrap3.mat']);
% unwrap=unwrap+(0.2/255)*randn(size(unwrap));
R_heng = unwrap;

sizeRhengpart=size(R_heng);
sizeLhengpart=size(L_heng);
mapu=nan(sizeRhengpart);
mapv=nan(sizeRhengpart);

nr=sizeRhengpart(1)*sizeRhengpart(2);

value_heng_Rdan=R_heng(:);
value_shu_Rdan=R_shu(:);
value_heng_Ldan=L_heng(:);
value_shu_Ldan=L_shu(:);

clear R_heng R_shu L_heng L_shu


minxwh=floor(min(value_heng_Ldan));%横条纹估计最低值
maxxwh=ceil(max(value_heng_Ldan));%横条纹估计最高值
ac=0.8;%配点精度
minxws=floor(min(value_shu_Ldan));
maxxws=ceil(max(value_shu_Ldan));
hn=10;
sn=10;
steph=ceil((maxxwh-minxwh)/hn);
steps=ceil((maxxws-minxws)/sn);
intfac=2;
% fid=fopen([calibpath 'xyzyesball.asc'],'wt');
fid2=fopen([path 'vumatch.asc'],'wt');
xyzmapx=nan(sizeRhengpart);
xyzmapy=nan(sizeRhengpart);
xyzmapz=nan(sizeRhengpart);

for time_j=minxws:steps:maxxws

    [ls]=find(value_shu_Ldan<=time_j+steps&value_shu_Ldan>time_j);
    [rs]=find(value_shu_Rdan<=time_j+steps+ac&value_shu_Rdan>time_j-ac);
    
    value_heng_L=value_heng_Ldan(ls); %第i行，col1到col2列的值(横)
    value_shu_L=value_shu_Ldan(ls);%第i行，col1到col2列的值(竖)
    value_heng_R=value_heng_Rdan(rs); %第i行，col1到col2列的值(横)
    value_shu_R=value_shu_Rdan(rs);%第i行，col1到col2列的值(竖)
    for time_j2=minxwh:steph:maxxwh
        [lh]=find(value_heng_L<=time_j2+steph&value_heng_L>time_j2);
        [rh]=find(value_heng_R<=time_j2+steph+ac&value_heng_R>time_j2-ac);
        if isempty(lh)||isempty(rh)
            continue
        end
        minus_sum=abs(repmat(value_heng_R(rh)',[length(lh),1])-repmat(value_heng_L(lh),[1,length(rh)]))+abs(repmat(value_shu_R(rh)',[length(lh),1])-repmat(value_shu_L(lh),[1,length(rh)]));
        [y,coor_min]=min(minus_sum,[],2);%找到每一hang最小值所在的索引
        [k]=find(y<ac);
        coor_min=rs(rh(coor_min));
        coor_min2=coor_min(k);
        coor_index2=ls(lh(k));
        [u1,v1]=ind2sub(sizeLhengpart,coor_index2);
        [u0,v0]=ind2sub(sizeRhengpart,coor_min2);
        
        if isempty(u1)
            continue
        end
        
        [hc]=find(u0==1|v0==1|u0==sizeLhengpart(1)|v0==sizeLhengpart(2));
        coor_min2(hc)=[];
        coor_index2(hc)=[];
        v0(hc)=[];
        u0(hc)=[];
        v1(hc)=[];
        u1(hc)=[];
        if isempty(u1)
            continue
        end
        sign_shu=sign(-value_shu_Rdan(coor_min2)+value_shu_Ldan(coor_index2));
        sign_heng=sign(-value_heng_Rdan(coor_min2)+value_heng_Ldan(coor_index2));
        coor_min2_inter_shu=coor_min2+sign_shu*sizeRhengpart(1);
        coor_min2_inter_heng=coor_min2+sign_heng;
        [hc]=find(isnan(coor_min2_inter_shu)|isnan(coor_min2_inter_heng));
        coor_min2(hc)=[];
        coor_index2(hc)=[];
        v0(hc)=[];
        u0(hc)=[];
        v1(hc)=[];
        u1(hc)=[];
        if isempty(u1)
            continue
        end
        fenmushu=value_shu_Rdan(coor_min2)-value_shu_Rdan(coor_min2_inter_shu);
        fenmuheng=value_heng_Rdan(coor_min2)-value_heng_Rdan(coor_min2_inter_heng);
        [hc]=find(abs(fenmushu)>intfac*ac|abs(fenmuheng)>intfac*ac|abs(fenmushu)==0|abs(fenmuheng)==0|isnan(fenmuheng)|isnan(fenmushu));
        coor_min2_inter_shu(hc)=[];
        coor_min2(hc)=[];
        sign_shu(hc)=[];
        coor_min2_inter_heng(hc)=[];
        sign_heng(hc)=[];
        coor_index2(hc)=[];
        v0(hc)=[];
        u0(hc)=[];
        v1(hc)=[];
        u1(hc)=[];
        fenmushu(hc)=[];
        fenmuheng(hc)=[];
        if isempty(u1)
            continue
        end
        v0=(value_shu_Rdan(coor_min2)-value_shu_Ldan(coor_index2)).*sign_shu./...
            (fenmushu)+v0;
        u0=(value_heng_Rdan(coor_min2)-value_heng_Ldan(coor_index2)).*sign_heng./...
            (fenmuheng)+u0;
        
        for i=1:length(u1)
            mapv(u1(i),v1(i))=v0(i);
            mapu(u1(i),v1(i))=u0(i);
        end
        
        vulr=[v1,u1,v0,u0];
        
        fprintf(fid2,'%e %e %e %e\n',vulr');
    end

end
% fclose(fid);
fclose(fid2);
save([path,'mapu.mat'],'mapu');%左相机对应的右相机像素行
save([path,'mapv.mat'],'mapv');%左相机对应的右相机像素列
% save([calibpath 'xyzmapx.mat'],'xyzmapx');
% save([calibpath 'xyzmapy.mat'],'xyzmapy');
% save([calibpath 'xyzmapz.mat'],'xyzmapz');
