function h=imshow4f(im,orient)
%
% h = imshow4f( imMag, orient)
%
% Generic 4D plotting function with image contrast control
% Modified from Joseph Cheng's imshow3s
%
% Inputs:
%     im      -   Image data
%     orient  -   Figure orientation, defaults to [1,2,3,4] (optional)
%
% Outputs:
%     h       -   Figure handle
%
% Example:
%     imshow3f( im , [-3,-2,1] );
%     flips 2nd and 3rd dimension and permutes input data to order [3,2,1]
%
% (c) Frank Ong 2013

im = squeeze(im);

ha = fighandle();

ha.figure = gcf;
clf; % clear current figure.

if (~isreal(sum(im(:)))  )
    im = abs(im);
end

if (nargin >= 2 && ~isempty(orient) && sum((orient~=[1,2,3])))
    perm = 1:length(size(im));
    perm(1:length(orient)) = abs(orient);
    im = permute(im,perm);
    
    for i = 1:length(orient)
        if (orient(i)<0)
            im = flipdim(im,i);
        end
    end
end

z = round(size(im,3)/2);
c = round(size(im,4)/2);
ha.singleSlice = 0;


%% Initialize contrast maps
ims = im(:,:,z,c);
map = [min(im(:)) max(im(:))]*1.0;
mapdiff = diff(map);
if mapdiff==0
    map = [map(1)-0.5,map(1)+0.5];
end

%% Setup figure and plot
figure(ha.figure); hold off;axis off;
axes('position', [0.11 0.11 0.7 0.8]);

ha.fh = imshow(ims,ha.map);

screen_size = get(0, 'ScreenSize');
figure_position = get(ha.figure, 'Position');
set(ha.figure, 'Position', [figure_position(1) figure_position(2) screen_size(3)/2 screen_size(4)/2])

ha.a = gca;
ha.z = z;
ha.c = c;
ha.map = map;
ha.busy = 0;


imshow3_plot(ha,im);
hslice = uicontrol(ha.figure,'Style','text',...
    'Units', 'Normalized',...
    'Position',[0.08 0.05 0.2 0.05],...
    'String',sprintf('Slice1: %d',z),'FontSize',14);

uicontrol(ha.figure,'Style', 'slider',...
    'SliderStep',[1/size(im,3) 1/size(im,3)],...
    'Min',1,'Max',size(im,3),'Value',z,...
    'Units', 'Normalized',...
    'Position', [0.08 0.00 0.2 0.05],...
    'Callback', {@int_slider_slice_callback,ha,hslice,im});


hslice2 = uicontrol(ha.figure,'Style','text',...
    'Units', 'Normalized',...
    'Position',[0.30 0.05 0.2 0.05],...
    'String',sprintf('Slice2: %d',z),'FontSize',14);

uicontrol(ha.figure,'Style', 'slider',...
    'SliderStep',[1/size(im,4) 1/size(im,4)],...
    'Min',1,'Max',size(im,4),'Value',c,...
    'Units', 'Normalized',...
    'Position', [0.30 0.00 0.2 0.05],...
    'Callback', {@int_slider_slice2_callback,ha,hslice2,im});


hmin = uicontrol(ha.figure,'Style','text',...
    'Units', 'Normalized',...
    'Position',[0.52 0.05 0.2 0.05],...
    'String',sprintf('Min WL: %.2f',map(1)),'FontSize',14);

uicontrol(ha.figure,'Style', 'slider',...
    'Min',map(1)-mapdiff,'Max',map(2)+mapdiff,'Value',map(1),...
    'Units', 'Normalized',...
    'Position', [0.52 0.00 0.2 0.05],...
    'Callback', {@int_slider_min_callback,ha,hmin,im});

hmax = uicontrol(ha.figure,'Style','text',...
    'Units', 'Normalized',...
    'Position',[0.74 0.05 0.2 0.05],...
    'String',sprintf('Max WL: %.2f',map(2)),'FontSize',14);

uicontrol(ha.figure,'Style', 'slider',...
    'Min',map(1)-mapdiff,'Max',map(2)+mapdiff,'Value',map(2),...
    'Units', 'Normalized',...
    'Position', [0.74 0.00 0.2 0.05],...
    'Callback', {@int_slider_max_callback,ha,hmax,im});



%% Setup output
if (nargout==1)
    h = ha.figure;
end


function int_slider_slice_callback(hslider,event,ha,ht,im)

z = round(get(hslider,'Value'));
set(hslider,'Value',z);
set(ht,'String',sprintf('Slice1: %d',z));
ha.z = z;
if ~ha.busy
    ha.busy = 1;
    imshow3_plot(ha,im);
    ha.busy = 0;
end


function int_slider_slice2_callback(hslider,event,ha,ht,im)

c = round(get(hslider,'Value'));
set(hslider,'Value',c);
set(ht,'String',sprintf('Slice2: %d',c));
ha.c = c;
if ~ha.busy
    ha.busy = 1;
    imshow3_plot(ha,im);
    ha.busy = 0;
end

function int_slider_max_callback(hslider,event,ha,ht,im)

m = get(hslider,'Value');
set(hslider,'Value',m);
set(ht,'String',sprintf('Max WL: %.2f',m));
ha.map = [min(ha.map(1),m-0.1) m];
if ~ha.busy
    ha.busy = 1;
    imshow3_plot(ha,im);
    ha.busy = 0;
end


function int_slider_min_callback(hslider,event,ha,ht,im)

m = get(hslider,'Value');
set(hslider,'Value',m);
set(ht,'String',sprintf('Min WL: %.2f',m));
ha.map = [m  max(ha.map(2),m+0.1)];
if ~ha.busy
    ha.busy = 1;
    imshow3_plot(ha,im);
    ha.busy = 0;
end



function imshow3_plot(ha,im)

z = ha.z;
c = ha.c;

ims = im(:,:,z,c);

set(ha.fh,'CData',ims);
set(ha.a, 'CLim',ha.map);
drawnow;

