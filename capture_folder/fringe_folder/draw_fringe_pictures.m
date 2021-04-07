m=1920;
n=1080;

% % 两列黑一列白

k=6;   
for s=1:k
    I1=zeros(n,m);
    for i=1:n
        for j=1:m/k
            t=k*j+s;
            if t>m
                t=t-m;
            end
            I1(i,t)=255;
            
            t=k*j+s+1;
            if t>m
                t=t-m;
            end
            I1(i,t)=255;
            
            t=k*j+s+2;
            if t>m
                t=t-m;
            end
            I1(i,t)=0;
            
            t=k*j+s+3;
            if t>m
                t=t-m;
            end
            I1(i,t)=0;
       %% I1(i,k*j+1)=255;
       %% I1(i,k*j+2)=255;
       %% I1(i,k*j+3)=255;
        end
    end
    I1=uint8(I1);
    imwrite(I1,['./' num2str(m) '_' num2str(n) '_' num2str(k) '_' num2str(s) '.png']);
end


%% 十字架

z=255*ones(1080,1920);
z(540:541,:)=0;
z(:,960:961)=0;
imagesc(z);

z=uint8(z);
imwrite(z,'./cross.jpg');



%% 正弦分布
m=1920;
n=1080;
T=4;
k=4;
for s=1:k
    I1=zeros(n,m);
    I2=zeros(n,m);
    for i=1:n
            I2(i,:)=127.5*(1+cos(2*pi*(i+T/k*(s-1)-n/2)/T));
    end
    I2=uint8(I2);
    imwrite(I2,['./h' num2str(s) '.png']);
    for j=1:m
            I1(:,j)=127.5*(1+cos(2*pi*(j+T/k*(s-1)-m/2)/T));
    end
    I1=uint8(I1);
    imwrite(I1,['./v' num2str(s) '.png']);
end



% % % 一黑一白
% % m=input('colomn=');
% % n=input('row=');
% % I1=zeros(n,m);
% % for i=1:n
% % for j=1:m
% % I1(i,j)=255*mod(j,2);
% % end
% % end
% % I1=unit8(I1);
% % colomn=1920
% % row=1200
% % imwrite(I1,['E:\matlab\1920_1200_2.bmp']);


% % I=zeros(2160,3840);
% % for i=1:540
% % for j=1:3840
% % I(4*i-1,j)=255;
% % I(4*i-2,j)=255;
% % end
% % end
% % I=uint8(I);
% % imwrite(I,['E:\matlab\3840_2160_4.bmp']);



% % 
% % I=zeros(2160,3840);
% % for i=1:360
% % for j=1:3840
% % I(6*i-1,j)=255;
% % I(6*i-2,j)=255;
% % I(6*i-3,j)=255;
% % end
% % end
% % I=uint8(I);
% % imwrite(I,['E:\matlab\3840_2160_6.bmp']);

m=1920;
n=1080;
T=4;
k=4;
a=[255 255 0 0;255 0 0 255;0 0 255 255;0 255 255 0];
for i=1:k
    I1=repmat(a(:,i),n/4,m);
    I1=uint8(I1);
    imwrite(I1,['./heng_' num2str(T) '_' num2str(m) '_' num2str(n) '_row' num2str(i) '.png']);
end


