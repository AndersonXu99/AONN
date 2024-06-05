%% 1920*1080 SLM 
%  This program is to generate 5 beams with weight a,b,c,d,e

% cd 'C:\Linear Operations\code\linear_iteration_1stSLM'

% size_real=[1920 1080]./Dim; % size_real = [160 90]
% Pattern=zeros(1080,1920);

% temp=zeros(1,Column*Row);
% temp(1:length(weight))=weight; %W1
% weight_shaped=reshape(temp,Column,Row);
% assigning the input W1 value to every element in the wegith_shaped matrix with the given row and col

% weight_shaped=flipud(weight_shaped);
% flips it upside down? why? 

% if time==0 
%   [Pattern_part,phi] = gsw_output(size_real, weight_shaped,interval); 
% else
%   [Pattern_part,phi] = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last,balance); % after the first iteration, we have a measured weight
% end

% %Pattern=repmat(Pattern_part,Dim(1),Dim(2));

% x=Dim(1)-1-mod(part-1,Dim(1));
% y=floor((part-1)/Dim(1));

% Pattern(y*size_real(2)+1:(y+1)*size_real(2),x*size_real(1)+1:(x+1)*size_real(1))=Pattern_part;


% if size(Pattern)~=[1080,1920]
% Pattern=Pattern';
% end

% Pattern=mod(Pattern+Correction,2*pi);
% %Pattern=Correction;


function [Image_SLM,phi] = gsw_output(size_real,weight,interval)
% size_real = [160 90]
% weight is the reshaped weight matrix

%% This function is to generate the gaussian beam according to the part and beam

%% This function is defined to calculate two type error:
%% if type==0, then the whole area is calculated. if type==1, then using
%% small area to calculate
%% not used? 


%% This function is defined to measure the simulation beam power
    function [power, power_sum]=IntensityMeasure(B, position)
        % interval is an input into the gsw_output algorithm
        rc=(position(:,2) - position(:,1) + 1) / interval;
        power=zeros(rc(1),rc(2));
        for i=1:rc(1)
            for ii=1:rc(2)
                x=position(1,1)+interval/2+(i-1)*interval;
                y=position(2,1)+interval/2+(ii-1)*interval;
                ratio_x=ceil(ratio*size_real(1)/size_real(2));
                power(i,ii)=sum(sum(abs(B(x-ratio_x/2:x+ratio_x/2-1,y-ratio/2:y+ratio/2-1)).^2));
            end
        end
        power_sum=sum(abs(B(:)).^2);
    end

%% This function is the algorithm for GS
    function [Output,g_next]=GS_algorithm(phase,g,position)
        %Step1: Gaussian beam with given phase    
        size_=(size_part-1)/2;
        [X,Y] = meshgrid(-size_(1):1:size_(1), -size_(2):1:size_(2));
        A0=exp( - ((X').^2)/(1000^2) - ((Y').^2)/(1000^2) ).*exp(1i*phase);

        B0=fftshift(fft2(A0,size_part(1),size_part(2)));
        A0=A0(real_rect(1,1):real_rect(1,2),real_rect(2,1):real_rect(2,2));
        %Step2: fft the beam
        
        B=fftshift(fft2(A0,size_part(1),size_part(2)));
        
        ak=sqrt(IntensityMeasure(B,position));
        
        g_next=(sqrt(weight)/sum(sqrt(weight(:))))./(ak/sum(ak(:))).*g;
        
        weight_next=g_next.*sqrt(weight);
        
        [at,~]=Multibeam(weight_next/mean(weight_next(:))*0.9);
        
        %Step3: Replace the amplitude by given intensity
        D=(at).*exp(1i*angle(B0));
        %Step4: ifft get new phase
        E=ifft2(ifftshift(D));
        Output=angle(E);
        
%         phase_small=zeros(size(phase));
%         phase_small(position(1,1):position(1,2),position(2,1):position(2,2))=phase(position(1,1):position(1,2),position(2,1):position(2,2));
%         A0_small=exp( - ((X').^2)/(500^2) - ((Y').^2)/(500^2) ).*exp(1i*phase_small);
%         B_small=fftshift(fft2(A0_small,size_part(1),size_part(2)));
%         power=sqrt(IntensityMeasure(B_small,position))
%         power.^2/sum(power(:).^2)-weight/sum(weight(:))
    end
%% This function is designed for generate pattern of arbitary multibeam
    function [Multipattern,position] = Multibeam(weight)
        % get the dimension of the reshaped weight matrix
        [row, column]=size(weight);
        
        % get the radius of a beam
        single_r=(interval-1)/2;
        % get the meshgrid of the beam
        [single_x,single_y]=meshgrid([-single_r:single_r],[-single_r:single_r]); % around 100 pixels
        % creates a 2D Gaussian function using the meshgrid that was just created. the value is greater the closer it is to the center of the meshgrid
        singlepattern=exp(-2*(single_x.^2+single_y.^2)/w0^2);
        % This line is using the repmat function to replicate the singlepattern matrix. The singlepattern matrix is replicated row times in the row dimension and column times in the column dimension. 
        % The output Multi is a larger matrix that consists of multiple copies of the singlepattern matrix arranged in a row by column grid. 
        % This is used to create a multi-beam pattern from the single beam pattern.
        Multi = repmat(singlepattern,row,column);

        % So, if Multi is a larger matrix and weight is a smaller matrix, 
        % this code is effectively applying a different weight to each interval x interval block of Multi, where the weights are given by the elements of weight.
        for i=1:row
            for ii=1:column
                Multi((i-1)*interval+ 1:i*interval, (ii-1)*interval + 1:(ii)*interval) = Multi((i-1)*interval + 1:i*interval, (ii-1) * interval + 1:(ii)*interval) * weight(i,ii);
            end
        end
        
%         e=length(weight(:))-sum(weight(:));
%         if e>0
%         [a,b]=size(Multi);
%         Multi(a/2-single_r:a/2+single_r,b/2-single_r:b/2+single_r)=Multi(a/2-single_r:a/2+single_r,b/2-single_r:b/2+single_r)+singlepattern*e;
%         end

        [Multi_x,Multi_y]=size(Multi);
        Multipattern=zeros(size_part); 
        % this gives the position ranges of the phase pattern
        position=[floor(size_part(1)/2) - floor(Multi_x/2)+1, floor(size_part(1)/2) + floor(Multi_x/2); floor(size_part(2)/2)-floor(Multi_y/2)+1, floor(size_part(2)/2)+floor(Multi_y/2)];

        Multipattern(position(1,1):position(1,2),position(2,1):position(2,2))=Multi;   
    end

% %% This function can generate modulation
%     function phase_mod=phase_mod(size,mod_T,mod_depth)
%         num=floor(size(1)/abs(mod_T));
%         phase_mod= mod_depth*pi*repmat(sin([1:mod_T]'/mod_T*2*pi),num,size(2));
%     end

%% Main function

tic
%size_real is the used area of SLMx

%parameter for target beam 
w0 = 1;
if size_real(1) > 500 
    ratio=2;
else size_real(1) < 500 % size_real(1) = 160 < 500
    ratio=4;
end
% ratio = 4 for our case

%interval defined before

size_part = [1,1] * size_real(1) * ratio;
% size_part = [640, 640]

padnum=(size_part-size_real) ./ 2;
% padnum = [640-160, 640-160] ./ 2 = [240, 240]

real_rect=[padnum(1)+1, padnum(1)+size_real(1); padnum(2)+1, padnum(2)+size_real(2)];
% real_rect = [241, 400; 241, 330]

[At,position] = Multibeam(sqrt(weight)); %intensity

%Phase0=angle(ifft2(ifftshift(I)));

Phase0=rand(size_part);

g=ones(size(weight));

[phi,~]=GS_algorithm(Phase0,g,position);
% [Error1(:,:,1),Ik1(:,:,1)]=ErrorCal(phi,position,0);
% [Error2(:,:,1),Ik2(:,:,1)]=ErrorCal(phi,position,1);
for nn=1:10
    [phi,g]=GS_algorithm(phi,g,position);
%    [Error1(:,:,nn+1),Ik1(:,:,nn+1)]=ErrorCal(phi,position,0);
%    [Error2(:,:,nn+1),Ik2(:,:,nn+1)]=ErrorCal(phi,position,1);
   
%    [At,~]=Multibeam(w0,interval,sqrt(g));
end


Phase_f=phi(real_rect(1,1):real_rect(1,2),real_rect(2,1):real_rect(2,2));
%Phase_grating=Phase_f+grating(fliplr(size_real),10)'+grating(size_real,-10);

%Phase_grating=Phase_f;%+phase_mod(size_real,5,0);

%Phase_all=zeros(size_SLM);

% Phase_all=Phase_f;
% %Phase_all(x+1:240+x,y+1:135+y)=Phase_grating;
% 
%  Bi=exp(1i*Phase_f);   
%  Bi=fftshift(fft2(Bi,size_part(1),size_part(2)));
% 
% % % % %powermeasure(abs(B))
% % % % 
%  I_real=uint8(abs(Bi)/max(max(abs(Bi)))*255);
%  imshow(I_real)
% % 

%normalize to pic
Phase_n=mod(Phase_f,2*3.1416);

% figure(6);
% imshow(I_real);
% 
% 
% figure(4);
% I=uint8(I*255);
% imshow(I);

% % figure(1);
Image_SLM=Phase_n';
% Image_SLM=uint8(Phase_n*255/(2*3.1416));
% Image_SLM=Image_SLM';
% imshow(Image_SLM);
toc

end

