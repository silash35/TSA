% Control of a distillation column subsystem (Alvarez et al, 2009) with MPC
%  basead on state-space model in the incremental form
clear
close all
clc
tic
s = tf('s'); % defining the Laplace's variables
Ts = 1     ; % sampling time [=] min

% Nominal model (used by MPC)
g = [2.3/s         -0.7/s 
     4.7/(9.3*s +1) 1.4/(6.8*s+1)] ; % transfer function matrix
io = 0*[0 0 0; 7 2 3]; % time delays associated with transfer function matrix
[ny,nu] = size(g);
gd = c2d(g,Ts,'zoh');

for i = 1:ny
    for j = 1:nu
        [num,den]=tfdata(gd(i,j),'v');
        delay = io(i,j);
        gd(i,j) = tf([zeros(1,delay/Ts) num],[den zeros(1,delay/Ts)],Ts);
    end
end
[Atil,Btil,Ctil,Dtil] = ssdata(gd) ;
[A,B,C]=immpc(Atil,Btil,Ctil);
% sysd = ss(Atil,Btil,Ctil,Dtil,Ts); % It is always necessary specify time sampling when discrete systems are considered here
% figure(1); step(sysd,80)
% hold on  ; step(g,80)

% Plant model
gp = [2.3*1.01/s           -0.7*1.02/s 
     1.1*4.7/(1.1*9.3*s +1) 1.05*1.4/(0.8*6.8*s+1)] ; % transfer function matrix
io = 0*[0 0 0; 7 2 3]; % time delays associated with transfer function matrix
[ny,nu] = size(gp);
gpd = c2d(gp,Ts,'zoh');

for i = 1:ny
    for j = 1:nu
        [num,den]=tfdata(gpd(i,j),'v');
        delay = io(i,j);
        gpd(i,j) = tf([zeros(1,delay/Ts) num],[den zeros(1,delay/Ts)],Ts);
    end
end

[Atilp,Btilp,Ctilp,Dtilp] = ssdata(gpd) ;
[Ap,Bp,Cp]=immpc(Atilp,Btilp,Ctilp);
% sysdp = ss(Atilp,Btilp,Ctilp,Dtilp,Ts); % It is always necessary specify time sampling when discrete systems are considered here
% figure(2); step(sysdp,80)
% hold on  ; step(gp,80)

% Updating the dimensions of variables
nu=size(g,2); % Number of manipulated inputs
ny=size(g,1); % Number of controlled outputs
nx=size(A,1); % Number of system states

% tuning parameters of MPC
p=120;% Output prediction horizon
m=3;% Input horizon
nsim=250;% Simulation time in sampling periods
q=[0.5,1];% Output weights
r=[.01,.01];% Input weights

umax=[10 10]';
umin=[0  0]';
dumax=[1 1]';

% Characteristics of process 
uss = [4.7 2.65]';
yss = [47 52.5]';

%  Defining the initial conditions (deviation variables)
xmk=zeros(nx,1); % It starts the steady-state
xpk=xmk;
ypk=Cp*xpk;
uk_1=uss-uss;

% State observer
Kf = FKalman(ny,A,C,100);

% Starting simulation
ysp=[];
for in=1:nsim
    uk(:,in)=uk_1+uss;
    yk(:,in)=ypk+yss;

    if in <= 30
        ys=[43 54]'; % Set-point of the outputs
    else
        ys=[43 54]';
    end
    
    [dukk,Vk]=issmpc(p,m,nu,ny,q,r,A,B,C,umax-uss,umin-uss,dumax,ys-yss,uk_1,xmk);
    duk=dukk(1:nu); % receding horizon
    Jk(in)=Vk; % control cost
    
    %Correction of the last control input
     xmk=A*xmk+B*duk;
     ymk=C*xmk;
  if in==100 
      xpk=Ap*xpk+Bp*(duk+0.2*[1 1]'); % inserting unmeasured disturbance into plant
%       xpk=Ap*xpk+Bp*duk;
      ypk=Cp*xpk; % plant measurement 
  else
      xpk=Ap*xpk+Bp*duk;
      ypk=Cp*xpk; % plant measurement 
  end
  
  %Correction of the last measurement
  de=ypk-ymk;
  xmk=xmk+Kf*de;
  uk_1=duk+uk_1;
  ysp=[ysp ys];
end

nc=size(yk,1);
figure(1)
for j=1:nc
    subplot(nc,1,j)
    plot(yk(j,:),'k-')
    hold on
    plot(ysp(j,:),'r--')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['y_' in];
    ylabel(yrot)
end
% legend('PV','set-point')

nc=size(uk,1);
figure(2)
for j=1:nc
subplot(nc,1,j)
    plot(uk(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['u_' in];
    ylabel(yrot)
end

figure(3)
plot(Jk)
xlabel('tempo nT')
ylabel('Cost function')

% data print
% ======================================
% outputs
% ======================================
y1 = [1:nsim;yk(1,:)]; 
y2 = [1:nsim;yk(2,:)];
ysp1 = [1:nsim;ysp(1,:)]; 
ysp2 = [1:nsim;ysp(2,:)];

[fid,msg] = fopen('y1.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',y1)
status = fclose(fid);

[fid,msg] = fopen('y2.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',y2)
status = fclose(fid);

[fid,msg] = fopen('ysp1.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',ysp1)
status = fclose(fid);

[fid,msg] = fopen('ysp2.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',ysp2)
status = fclose(fid);
% ======================================

% ======================================
% inputs
% ======================================
u1 = [1:nsim;uk(1,:)]; 
u2 = [1:nsim;uk(2,:)];
% u3 = [1:nsim;uk(3,:)];

[fid,msg] = fopen('u1.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',u1)
status = fclose(fid);

[fid,msg] = fopen('u2.dat','w');
fprintf(fid, '%6.3f  %6.3f\n',u2)
status = fclose(fid);

% [fid,msg] = fopen('u3.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',u3)
% status = fclose(fid);
% ======================================

% ======================================
% cost
% ======================================
% V = [1:nsim;Jk(1,:)]; 
% [fid,msg] = fopen('V.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',V)
% status = fclose(fid);

