%% 1920*1080 SLM 
%  This program is to generate 5 beams with weight a,b,c,d,e

function [Image_SLM,phase] = gs_iteration_modified(size_real,weight,interval,phi,e)
    function Output=GS_algorithm(phase,weight)
    %Step1: Gaussian beam with given phase    
        size_=(size_part-1)/2;
        [X,Y] = meshgrid(-size_(1):1:size_(1), -size_(2):1:size_(2));
        A0=exp( - ((X').^2)/(1000^2) - ((Y').^2)/(1000^2) ).*exp(1i*phase);
        B0=fftshift(fft2(A0,size_part(1),size_part(2)));
        %Step2: fft the beam      
        [at,~]=Multibeam(sqrt(weight));       
        %Step3: Replace the amplitude by given intensity
        D=(at).*exp(1i*angle(B0));
        %Step4: ifft get new phase
        E=ifft2(ifftshift(D));
        Output=angle(E);
    end
%% This function is designed for generate pattern of arbitary multibeam
    function [Multipattern,position]=Multibeam(weight)
        [row, column]=size(weight);
        
        single_r=(interval-1)/2;
        [single_x,single_y]=meshgrid([-single_r:single_r],[-single_r:single_r]);
        singlepattern=exp(-2*(single_x.^2+single_y.^2)/w0^2);
        Multi=repmat(singlepattern,row,column);

        for i=1:row
            for ii=1:column
                Multi((i-1)*interval+1:i*interval,(ii-1)*interval+1:(ii)*interval)=Multi((i-1)*interval+1:i*interval,(ii-1)*interval+1:(ii)*interval)*weight(i,ii);
            end
        end     
        
%        e=length(weight(:))-length(find(weight==0))-sum(weight(:));
%        if e>0
%        [a,b]=size(Multi);
%        Multi(a/2-single_r:a/2+single_r,b/2-single_r:b/2+single_r)=Multi(a/2-single_r:a/2+single_r,b/2-single_r:b/2+single_r)+singlepattern*e;
%        end
        
        
        [Multi_x,Multi_y]=size(Multi);
        Multipattern=zeros(size_part);       
        position=[floor(size_part(1)/2)-floor(Multi_x/2)+1,floor(size_part(1)/2)+floor(Multi_x/2);floor(size_part(2)/2)-floor(Multi_y/2)+1,floor(size_part(2)/2)+floor(Multi_y/2)];
        Multipattern(position(1,1):position(1,2),position(2,1):position(2,2))=Multi;
        Multipattern(position(1,1):position(1,2),position(2,1):position(2,2))=Multi;
        if e>0
        Multipattern(position(1,1)-size(singlepattern,1)+1:position(1,1),size(Multipattern,1)/2-floor(size(singlepattern,1)/2)+1:size(Multipattern,1)/2-floor(size(singlepattern,1)/2)+size(singlepattern,1))=singlepattern*e;
    
        end
    end

%% Main function

tic
%size_real is the used area of SLMx

%parameter for target beam 
w0=1;
if size_real(1)>500
    ratio=2;
else size_real(1)<500
    ratio=4;
end
%interval defined before
size_part=[1 1]*size_real(1)*ratio;
padnum=(size_part-size_real)./2;
real_rect=[padnum(1)+1,padnum(1)+size_real(1);padnum(2)+1,padnum(2)+size_real(2)];


phase=GS_algorithm(phi,weight);

Phase_f=phase(real_rect(1,1):real_rect(1,2),real_rect(2,1):real_rect(2,2));

%normalize to pic
Phase_n=mod(Phase_f,2*3.1416);

Image_SLM=Phase_n';

toc

end

