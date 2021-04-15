clear
close all
N = 512;
[x,y]=meshgrid(1:N);
shape = zeros(N,N);
phaseChange = 10;
shape(:,300:end) = phaseChange;
x1=meshgrid(1:100);
shape(1:100,200:299) = x1*phaseChange/100;
shape(413:512,200:299) = x1*phaseChange/100;
image1 = shape + 0.05*x + 0.002*y;
figure, colormap(gray(256)), imagesc(image1)
title('Original image displayed as a visual intensity array')
xlabel('Pixels'), ylabel('Pixels')

figure 
surf(image1,'FaceColor','interp', 'EdgeColor','none', 'FaceLighting','phong')
view(-30,30), camlight left, axis tight
title('Original image displayed as a surface plot')
xlabel('Pixels'), ylabel('Pixels'), zlabel('Phase in radians')

image1_wrapped = atan2(sin(image1), cos(image1));
figure, colormap(gray(256)), imagesc(image1_wrapped)
title('Wrapped image displayed as a visual intensity array')

image1_unwrapped = image1_wrapped;
for i=1   % 先unwrap row
 image1_unwrapped(i,:) = unwrap(image1_unwrapped(i,:));
end
%Then unwrap all the columns one at a time
for i=1:N
 image1_unwrapped(:,i) = unwrap(image1_unwrapped(:,i));
end
figure 
surf(image1_unwrapped,'FaceColor','interp', 'EdgeColor','none',...
'FaceLighting','phong')
view(-30,30), camlight left, axis tight
title('Unwrapped phase map using the Itoh unwrapper: the first method')
xlabel('Pixels'), ylabel('Pixels'), zlabel('Phase in radians')
figure, colormap(gray(256)), imagesc(image1_unwrapped)
title('Unwrapped image using the Itoh algorithm: the first method')
xlabel('Pixels'), ylabel('Pixels')

function x = haha()
for i = 1:10    % 这种模式做10个
    t = clock; t=num2str(t); t=strrep(t,' ',''); % 获取当前时间并转换为string
    image_t = 2*peaks(N) + 0.1*x*rand(1) + 0.01*y*rand(1);   % 原始解包数据,y_truth
    save([t '_' num2str(i) '.mat'],'image_t');
    image1_wrapped = atan2(sin(image_t), cos(image_t));
    for noise_variance = 0:0.01:0.4
        image_noise = image_t + noise_variance*randn(N,N);%不存
        imagen_wrapped = atan2(sin(image_noise), cos(image_noise));% x,包裹
        save([t '_' num2str(i) '_' num2str(noise_variance)...
            '.mat'],'imagen_wrapped');
        % colormap(gray(256)), imagesc(imagen_wrapped)
    end
end
end