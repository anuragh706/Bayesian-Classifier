

imgs1=cell(1,200);      %% initial image cells for 3 different classes for 128 pixel image
imgs2=cell(1,200);
imgs3=cell(1,200);

B1=cell(1,200);      %% image cells for 3 different class for 32 X 32 (resized)
B2=cell(1,200);
B3=cell(1,200);
for i=1:200       %%to read the images into imgs
    
    a1='G:\PRML\lecture\Assignment_list\TrainCharacters\1\';     %% strcat used for setting the path
    s1=strcat(a1,num2str(i),'.jpg');
    imgs1{i}=imread(s1);                                           %% to read the image  
    
    
    a2='G:\PRML\lecture\Assignment_list\TrainCharacters\2\';
    s2=strcat(a2,num2str(i),'.jpg');
    imgs2{i}=imread(s2);
    
    a3='G:\PRML\lecture\Assignment_list\TrainCharacters\3\';
    s3=strcat(a3,num2str(i),'.jpg');
    imgs3{i}=imread(s3);
    
end

for j=1:200   %% resizing images
    
    B1{j}=imresize(imgs1{j},0.25);                       %% image resizing function
    B2{j}=imresize(imgs2{j},0.25);
     B3{j}=imresize(imgs3{j},0.25);
    
end

g=1;
for  k=1:200                                              %%creating a feature vector
    
    for p=1:32
        
        
        for m=1:32
            
        Feat1(k,g)=B1{k}(p,m);
        Feat2(k,g)=B2{k}(p,m);
        Feat3(k,g)=B3{k}(p,m);
        g=g+1;
        
        end
        
    end
    g=1;
    
end

mu1=mean(Feat1);                     %% calculating mu vector
mu1=mu1';
Feat1=double(Feat1);
sigma1=cov(Feat1);                              %% covariance matrix
sigma1=sigma1+0.38288*eye(1024);                 %% Covariance matrix altered using regularization parameter
sigma1inv=inv(sigma1);                          %% inverse of covariance matrix

mu2=mean(Feat2);                               
mu2=mu2';
Feat2=double(Feat2);
sigma2=cov(Feat2);
sigma2=sigma2+0.3785*eye(1024);
sigma2inv=inv(sigma2);

mu3=mean(Feat3); 
mu3=mu3';
Feat3=double(Feat3);
sigma3=cov(Feat3);
sigma3=sigma3+0.5095*eye(1024);
sigma3inv=inv(sigma3);

Feat4=[Feat1;Feat2];
Feat5=[Feat4;Feat3];
sigma5=cov(Feat5);
sigma5=sigma5 + 0.24815*eye(1024);
sigma5inv=inv(sigma5);

sigma6=eye(1024);

for i=201:300       %%to read the test images into imgs
    
    ts1='G:\PRML\lecture\Assignment_list\TestCharacters\TestCharacters\1\';     %% strcat used for setting the path
    as1=strcat(ts1,num2str(i),'.jpg');
    test1{i-200}=imread(as1);                                           %% to read the image  
    
    
    ts2='G:\PRML\lecture\Assignment_list\TestCharacters\TestCharacters\2\';     %% strcat used for setting the path
    as2=strcat(ts2,num2str(i),'.jpg');
    test2{i-200}=imread(as2);  
    
    ts3='G:\PRML\lecture\Assignment_list\TestCharacters\TestCharacters\3\';
    as3=strcat(ts3,num2str(i),'.jpg');
    test3{i-200}=imread(as3);
    
end

for j=1:100   %% resizing images
    
    testB1{j}=imresize(test1{j},0.25);                       %% image resizing function
    testB2{j}=imresize(test2{j},0.25);
     testB3{j}=imresize(test3{j},0.25);
    
end

g=1;
for  k=1:100                                              %%creating a feature vector
    
    for p=1:32
        
        
        for m=1:32
            
        testFeat1(k,g)=testB1{k}(p,m);
        testFeat2(k,g)=testB2{k}(p,m);
        testFeat3(k,g)=testB3{k}(p,m);
        g=g+1;
        
        end
        
    end
    g=1;
    
end

testFeat1=double(testFeat1);
testFeat2=double(testFeat2);
testFeat3=double(testFeat3);

%% for part 1
for l=1:100

    
x1=[-0.5*((testFeat1(l:l,1:1024))'-mu1)'*(sigma1inv)*((testFeat1(l:l,1:1024))'-mu1)]-0.5*log(det(sigma1))/log(2.71828);
x2=[-0.5*((testFeat1(l:l,1:1024))'-mu2)'*(sigma2inv)*((testFeat1(l:l,1:1024))'-mu2)]-0.5*log(det(sigma2))/log(2.71828);
x3=[-0.5*((testFeat1(l:l,1:1024))'-mu3)'*(sigma3inv)*((testFeat1(l:l,1:1024))'-mu3)]-0.5*log(det(sigma3))/log(2.71828);

if(x1>x2 && x1>x3)
class1(l,1)=1;
end

if(x2>x1 && x2>x3)
   class1(l,1)=2; 
end

if(x3>x2 && x3 > x1)
   class1(l,1)=3; 
end

y1=[-0.5*((testFeat2(l:l,1:1024))'-mu1)'*(sigma1inv)*((testFeat2(l:l,1:1024))'-mu1)]-0.5*log(det(sigma1))/log(2.71828);
y2=[-0.5*((testFeat2(l:l,1:1024))'-mu2)'*(sigma2inv)*((testFeat2(l:l,1:1024))'-mu2)]-0.5*log(det(sigma2))/log(2.71828);
y3=[-0.5*((testFeat2(l:l,1:1024))'-mu3)'*(sigma3inv)*((testFeat2(l:l,1:1024))'-mu3)]-0.5*log(det(sigma3))/log(2.71828);

if(y1>y2 && y1>y3)
class2(l,1)=1;
end

if(y2>y1 && y2>y3)
   class2(l,1)=2; 
end

if(y3>y2 && y3 > y1)
   class2(l,1)=3; 
end

z1=[-0.5*((testFeat3(l:l,1:1024))'-mu1)'*(sigma1inv)*((testFeat3(l:l,1:1024))'-mu1)]-0.5*log(det(sigma1))/log(2.71828);
z2=[-0.5*((testFeat3(l:l,1:1024))'-mu2)'*(sigma2inv)*((testFeat3(l:l,1:1024))'-mu2)]-0.5*log(det(sigma2))/log(2.71828);
z3=[-0.5*((testFeat3(l:l,1:1024))'-mu3)'*(sigma3inv)*((testFeat3(l:l,1:1024))'-mu3)]-0.5*log(det(sigma3))/log(2.71828);

if(z1>z2 && z1>z3)
class3(l,1)=1;
end

if(z2>z1 && z2>z3)
   class3(l,1)=2; 
end

if(z3>z2 && z3 > z1)
   class3(l,1)=3; 
end



end

c1=0;
c2=0;
c3=0;
for p=1:100
    
    
if(class1(p,1)==1)    
c1=c1+1;
end

if(class2(p,1)==2)    
c2=c2+1;
end

if(class3(p,1)==3)    
c3=c3+1;
end

end


%% for part 2 
for l=1:100

    
x1=[-0.5*((testFeat1(l:l,1:1024))'-mu1)'*(sigma5inv)*((testFeat1(l:l,1:1024))'-mu1)]-0.5*log(det(sigma5))/log(2.71828);
x2=[-0.5*((testFeat1(l:l,1:1024))'-mu2)'*(sigma5inv)*((testFeat1(l:l,1:1024))'-mu2)]-0.5*log(det(sigma5))/log(2.71828);
x3=[-0.5*((testFeat1(l:l,1:1024))'-mu3)'*(sigma5inv)*((testFeat1(l:l,1:1024))'-mu3)]-0.5*log(det(sigma5))/log(2.71828);

if(x1>x2 && x1>x3)
class21(l,1)=1;
end

if(x2>x1 && x2>x3)
   class21(l,1)=2; 
end

if(x3>x2 && x3 > x1)
   class21(l,1)=3; 
end

y1=[-0.5*((testFeat2(l:l,1:1024))'-mu1)'*(sigma5inv)*((testFeat2(l:l,1:1024))'-mu1)]-0.5*log(det(sigma5))/log(2.71828);
y2=[-0.5*((testFeat2(l:l,1:1024))'-mu2)'*(sigma5inv)*((testFeat2(l:l,1:1024))'-mu2)]-0.5*log(det(sigma5))/log(2.71828);
y3=[-0.5*((testFeat2(l:l,1:1024))'-mu3)'*(sigma5inv)*((testFeat2(l:l,1:1024))'-mu3)]-0.5*log(det(sigma5))/log(2.71828);

if(y1>y2 && y1>y3)
class22(l,1)=1;
end

if(y2>y1 && y2>y3)
   class22(l,1)=2; 
end

if(y3>y2 && y3 > y1)
   class22(l,1)=3; 
end

z1=[-0.5*((testFeat3(l:l,1:1024))'-mu1)'*(sigma5inv)*((testFeat3(l:l,1:1024))'-mu1)]-0.5*log(det(sigma5))/log(2.71828);
z2=[-0.5*((testFeat3(l:l,1:1024))'-mu2)'*(sigma5inv)*((testFeat3(l:l,1:1024))'-mu2)]-0.5*log(det(sigma5))/log(2.71828);
z3=[-0.5*((testFeat3(l:l,1:1024))'-mu3)'*(sigma5inv)*((testFeat3(l:l,1:1024))'-mu3)]-0.5*log(det(sigma5))/log(2.71828);

if(z1>z2 && z1>z3)
class23(l,1)=1;
end

if(z2>z1 && z2>z3)
   class23(l,1)=2; 
end

if(z3>z2 && z3 > z1)
   class23(l,1)=3; 
end



end


c21=0;
c22=0;
c23=0;
for p=1:100
    
    
if(class21(p,1)==1)    
c21=c21+1;
end

if(class22(p,1)==2)    
c22=c22+1;
end

if(class23(p,1)==3)    
c23=c23+1;
end

end


%% for part 3

for l=1:100

    
x1=[-0.5*((testFeat1(l:l,1:1024))'-mu1)'*(sigma6)*((testFeat1(l:l,1:1024))'-mu1)]-0.5*log(det(sigma6))/log(2.71828);
x2=[-0.5*((testFeat1(l:l,1:1024))'-mu2)'*(sigma6)*((testFeat1(l:l,1:1024))'-mu2)]-0.5*log(det(sigma6))/log(2.71828);
x3=[-0.5*((testFeat1(l:l,1:1024))'-mu3)'*(sigma6)*((testFeat1(l:l,1:1024))'-mu3)]-0.5*log(det(sigma6))/log(2.71828);

if(x1>x2 && x1>x3)
class31(l,1)=1;
end

if(x2>x1 && x2>x3)
   class31(l,1)=2; 
end

if(x3>x2 && x3 > x1)
   class31(l,1)=3; 
end

y1=[-0.5*((testFeat2(l:l,1:1024))'-mu1)'*(sigma6)*((testFeat2(l:l,1:1024))'-mu1)]-0.5*log(det(sigma6))/log(2.71828);
y2=[-0.5*((testFeat2(l:l,1:1024))'-mu2)'*(sigma6)*((testFeat2(l:l,1:1024))'-mu2)]-0.5*log(det(sigma6))/log(2.71828);
y3=[-0.5*((testFeat2(l:l,1:1024))'-mu3)'*(sigma6)*((testFeat2(l:l,1:1024))'-mu3)]-0.5*log(det(sigma6))/log(2.71828);

if(y1>y2 && y1>y3)
class32(l,1)=1;
end

if(y2>y1 && y2>y3)
   class32(l,1)=2; 
end

if(y3>y2 && y3 > y1)
   class32(l,1)=3; 
end

z1=[-0.5*((testFeat3(l:l,1:1024))'-mu1)'*(sigma6)*((testFeat3(l:l,1:1024))'-mu1)]-0.5*log(det(sigma6))/log(2.71828);
z2=[-0.5*((testFeat3(l:l,1:1024))'-mu2)'*(sigma6)*((testFeat3(l:l,1:1024))'-mu2)]-0.5*log(det(sigma6))/log(2.71828);
z3=[-0.5*((testFeat3(l:l,1:1024))'-mu3)'*(sigma6)*((testFeat3(l:l,1:1024))'-mu3)]-0.5*log(det(sigma6))/log(2.71828);

if(z1>z2 && z1>z3)
class33(l,1)=1;
end

if(z2>z1 && z2>z3)
   class33(l,1)=2; 
end

if(z3>z2 && z3 > z1)
   class33(l,1)=3; 
end



end

c31=0;
c32=0;
c33=0;
for p=1:100
    
    
if(class31(p,1)==1)    
c31=c31+1;
end

if(class32(p,1)==2)    
c32=c32+1;
end

if(class33(p,1)==3)    
c33=c33+1;
end

end


avgacc=(c1+c2+c3+c21+c22+c23+c31+c32+c33)/9;
