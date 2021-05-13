function [phase_unwrap,phase_calibrate,N_iteration,t]=CPULSI(phase_input,MASK_input,...
    Nmax_iteration,Error,XC,YC,Calibration)
%CPULSI: Phase unwrapping based on least-squares, iteration and calibration
%        to phase derivatives
%        The algorithm is described in the papers:
%        [1] Haiting XIA,Silvio MONTRESOR,Rongxin GUO,Junchang LI,Feng YAN,Heming CHENG,AND Pascal PICART,"Phase
%        calibration unwrapping algorithm for phase data corrupted by
%        strong decorrelation speckle noise". Opt. Express 24(25), 28713-28730 (2016) (for CPULSI)
%        [2] Haiting XIA,Rongxin GUO, and Zebin FAN etc..
%        "Non-invasive Mechanical Measurement for Transparent Objects by Digital Holographic Interferometry 
%        Based on Iterative Least-Squares Phase Unwrapping". Exp. Mech. 52(4), 439-445(2012) (for PULSI)
%        last change: Haiting XIA, 8/12/2016  
%        Email: htxia2006@163.com
%INPUT:  phase_input is the wrapped phase to be unwrapped, double, [-pi,pi]
%        MASK_input is a mask matrix which use 0 to represent the pixels in
%              masked region and 1 to represent the other pixels
%        Nmax_iteration is the max numbers of iterations
%        Error is the threshold of the unwrapped phase error when iteration
%              is stopped
%        [XC,YC] a phase-known point
%        Calibration: option parameter to calibration, 'true' for CPULSI
%        (using calibration to derivatives) used for noisy wrapped phase, 'false' for PULSI (non
%        calibration) used for noise-free (or denoised) wrapped phase
%OUTPUT: phase_unwrap is least-squares unwrapped phase
%        phase_calibrate is calibrated unwrapped phase
%        N_iteration is the number of iterations used finally
%        t is the computing time used
[m,n]=size(phase_input);
C13=zeros(m,n);% set initial unwrapped phase
C4=phase_input.*MASK_input;% set inputed wrapped phase as initial unwrapped phase error to be unwrapped
tic;
for i=1:Nmax_iteration
%     disp(num2str(i));
    C41=C4;
    C8=CPULS(C4,MASK_input,Calibration);% phase unwrapping to unwrapped phase errors by CPULS
    C13=C13+C8;% unwrapped phase
    C13=C13+mean2(phase_input(max(1,XC-3):min(m,XC+3),max(1,YC-3):min(n,YC+3)))-mean2(C13(max(1,XC-3):min(m,XC+3),max(1,YC-3):min(n,YC+3)));
    C10=phase_input+2*pi*round((C13-phase_input)/(2*pi));% calibration to the unwrapped phase by initial wrapped phase
    C4=(C10-C13).*MASK_input;% calculate the unwrapped phase error
    if mean2(abs(C4-C41))<Error
        break;
    end
end
t=toc;% compute the time of iterations
N_iteration=i;
phase_unwrap=C13;
phase_calibrate=C10;

function phase_unwrapped=CPULS(phase_wrapped,MASK,Calibration)
%CPULS: phase unwrapping algorithm based on the least-squares and
%       calibration to the phase derivatives
CC=phase_wrapped.*MASK;
[x,y]=size(CC);
DX=zeros(x,y+1);DY=zeros(x+1,y);
DX(:,2:y)=CC(:,2:y)-CC(:,1:(y-1));
DX=angle(exp(1i*DX));
if Calibration
Gx=abs(mean2(DX));
Thx=std2(DX);
DX(DX>=Thx)=Gx;
DX(DX<=-Thx)=-Gx;
end
DX(:,1:y)=DX(:,1:y).*MASK;
DX(:,2:(y+1))=DX(:,2:(y+1)).*MASK;
% Compute the phase differences with respect to j
DY(2:x,:)=CC(2:x,:)-CC(1:(x-1),:);
DY=angle(exp(1i*DY));
if Calibration
Gy=abs(mean2(DY));
Thy=std2(DY);
DY(DY>=Thy)=Gy;
DY(DY<=-Thy)=-Gy;
end
DY(1:x,:)=DY(1:x,:).*MASK;
DY(2:(x+1),:)=DY(2:(x+1),:).*MASK;
C5=(DX(:,2:(y+1))-DX(:,1:y))+(DY(2:(x+1),:)-DY(1:x,:));%¦Ñij
C6=dct2(C5);%  ^¦Ñij, the DCT of ¦Ñij
%compute the DCT values ^¦Õij of phase_unwrapped ¦Õij:
C7=zeros(size(C6));
for m=1:x
    for n=1:y
        if ((m==1) && (n==1))
            C7(m,n)=C6(m,n);
        else
            C7(m,n)=C6(m,n)/(2*(cos(pi*(m-1)/x)+cos(pi*(n-1)/y)-2));
        end
    end
end
phase_unwrapped=idct2(C7);%perform IDCT of ^¦Õij to obtain phase_unwrapped ¦Õij
