classdef SiTraNo_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                     matlab.ui.Figure
        GridLayout                   matlab.ui.container.GridLayout
        LeftPanel                    matlab.ui.container.Panel
        GaugeN                       matlab.ui.control.SemicircularGauge
        TabGroup                     matlab.ui.container.TabGroup
        SpectrumTab                  matlab.ui.container.Tab
        SpectrumAnalyzerLabel        matlab.ui.control.Label
        Spectrum                     matlab.ui.control.UIAxes
        TonalTab                     matlab.ui.container.Tab
        STFTTonalComponentLabel      matlab.ui.control.Label
        STFT_S                       matlab.ui.control.UIAxes
        TransientTab                 matlab.ui.container.Tab
        STFTTransientComponentLabel  matlab.ui.control.Label
        STFT_T                       matlab.ui.control.UIAxes
        NoiseTab                     matlab.ui.container.Tab
        STFTNoiseComponentLabel      matlab.ui.control.Label
        STFT_N                       matlab.ui.control.UIAxes
        EvaluationTab                matlab.ui.container.Tab
        ESLabel                      matlab.ui.control.Label
        BinswArtifactsLabel          matlab.ui.control.Label
        STonalLabel                  matlab.ui.control.Label
        TTransientLabel              matlab.ui.control.Label
        NNoiseLabel                  matlab.ui.control.Label
        EnergyLabel_2                matlab.ui.control.Label
        ASLabel                      matlab.ui.control.Label
        ETLabel                      matlab.ui.control.Label
        ATLabel                      matlab.ui.control.Label
        ENLabel                      matlab.ui.control.Label
        ANLabel                      matlab.ui.control.Label
        ArtifactsperTimeFrameLabel   matlab.ui.control.Label
        ArtifactsAxes                matlab.ui.control.UIAxes
        GaugeT                       matlab.ui.control.SemicircularGauge
        GaugeS                       matlab.ui.control.SemicircularGauge
        TLabel                       matlab.ui.control.Label
        SLabel                       matlab.ui.control.Label
        NLabel                       matlab.ui.control.Label
        UIAxes2                      matlab.ui.control.UIAxes
        RightPanel                   matlab.ui.container.Panel
        activeLabel                  matlab.ui.control.Label
        ParametersLabel              matlab.ui.control.Label
        STOPButton                   matlab.ui.control.StateButton
        PLAYButton                   matlab.ui.control.StateButton
        SiTraNoLabel                 matlab.ui.control.Label
        TonesCheckBox                matlab.ui.control.CheckBox
        LoopCheckBox                 matlab.ui.control.CheckBox
        NoiseCheckBox                matlab.ui.control.CheckBox
        TransientsCheckBox           matlab.ui.control.CheckBox
        OpenButton                   matlab.ui.control.Button
        TransientsSlider             matlab.ui.control.Slider
        Label_2                      matlab.ui.control.Label
        NoiseSlider                  matlab.ui.control.Slider
        Label                        matlab.ui.control.Label
        TonesSlider                  matlab.ui.control.Slider
        Label_3                      matlab.ui.control.Label
        MethodDropDown               matlab.ui.control.DropDown
        MethodDropDownLabel          matlab.ui.control.Label
        StatusLamp                   matlab.ui.control.Lamp
        StatusLabel                  matlab.ui.control.Label
        NFFTEditField                matlab.ui.control.NumericEditField
        NFFTLabel                    matlab.ui.control.Label
        ThresholdSEditField          matlab.ui.control.NumericEditField
        ThresholdSEditFieldLabel     matlab.ui.control.Label
        ThresholdTEditField          matlab.ui.control.NumericEditField
        ThresholdTEditFieldLabel     matlab.ui.control.Label
        UpdatePlotsButton            matlab.ui.control.Button
        SaveButton                   matlab.ui.control.Button
        Image                        matlab.ui.control.Image
        EditMenu                     matlab.ui.container.Menu
        GlobalparametersMenu         matlab.ui.container.Menu
        FileMenu                     matlab.ui.container.Menu
        OpenMenu                     matlab.ui.container.Menu
        SaveMenu                     matlab.ui.container.Menu
        InfoMenu                     matlab.ui.container.Menu
        VersionLogMenu               matlab.ui.container.Menu
        AboutMenu                    matlab.ui.container.Menu
        WebsiteMenu                  matlab.ui.container.Menu
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    
    properties (Access = private)
        Audio %audioblock
        Length % length
        nWin % nWin
        win % analysis window
        nHop % hopsize
        NFFT % NFFT
        params % params
        nMedianH % nmg
        nMedianV % nmv
        Player % audioplayer
        Position % Position
        nS % integration S
        nT % integration T
        nN % integration N
        FFTsize % FFTsize
        frq % frq
        FigurePosition %esplicativo
        Threshold % same
    end
    
    methods (Access = private)
        
        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        function myUpdateAppLayout(app, event)
            %currentFigureWidth = app.UIFigure.Position(3);
            
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnWidth = {'1x', 299};
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;
            
%             if(currentFigureWidth <= app.onePanelWidth)
%                 % Change to a 2x1 grid
%                 app.GridLayout.RowHeight = {480, 480};
%                 app.GridLayout.ColumnWidth = {'1x'};
%                 app.RightPanel.Layout.Row = 2;
%                 app.RightPanel.Layout.Column = 1;
%             else
%                 % Change to a 1x2 grid
%                 
%             end   
            
         end
        
        
        function resetInitialValues(app)
            app.TonesCheckBox.Value = 1;
            app.TonesSlider.Value = 100;
            app.TransientsCheckBox.Value = 1;
            app.TransientsSlider.Value = 100;
            app.NoiseCheckBox.Value = 1;
            app.NoiseSlider.Value = 100;
            app.StatusLamp.Color = 'g';
            app.activeLabel.Text = "(Active)";
            %app.MethodDropDown.Value = "HPR";
            
            app.params.beta = 2.5;
            app.params.c = 0.2; 
            app.params.r = [10000 10000]; 
            app.params.sgm = 2*[1.4 1.4];
            app.params.tol = 20;
            
            app.nS = 2047; app.nT = 255; app.nN = 511;
                        
            app.nWin.r0 = 2048; app.nHop.r0 = app.nWin.r0 / 8;
            app.NFFT.r0 = app.nWin.r0; app.win.r0 = hann(app.nWin.r0,'periodic');
            app.nWin.r1 = 8192; app.nHop.r1 = app.nWin.r1 / 8;
            app.NFFT.r1 = app.nWin.r1; app.win.r1 = hann(app.nWin.r1,'periodic');
            app.nWin.r2 = 512; app.nHop.r2 = app.nWin.r2 / 8;
            app.NFFT.r2 = app.nWin.r2; app.win.r2 = hann(app.nWin.r2,'periodic');
            
            app.FFTsize = 2048; app.NFFTEditField.Value = app.FFTsize;
            app.Threshold.S = -89;
            app.ThresholdSEditField.Value = app.Threshold.S;
            app.Threshold.T = -85;
            app.ThresholdTEditField.Value = app.Threshold.T;
            %app.frq = ([1:app.NFFT.r0/2]) * app.Audio.fs/2;
            
            app.nMedianH.r0 = round(200e-3 * app.Audio.fs / app.nHop.r0 );
            app.nMedianV.r0 = round(500 * app.NFFT.r0 / app.Audio.fs);
            app.nMedianH.r1 = round(200e-3 * app.Audio.fs / app.nHop.r1 );
            app.nMedianV.r1 = round(500 * app.NFFT.r1 / app.Audio.fs);
            app.nMedianH.r2 = round(200e-3 * app.Audio.fs / app.nHop.r2 );
            app.nMedianV.r2 = round(500 * app.NFFT.r2 / app.Audio.fs);
            
            
            
        end
        
        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
        function [Y,T] = stft(app,x,win,nHop,NFFT)
            nWin = length(win);
            L = length(x);
            
            nFrames = floor((L-nWin)/nHop+1);
            nBins = NFFT/2+1;
            Y = zeros(nBins,nFrames);
            T = zeros(1,nFrames);
            
            pin = 0;
            x = [x;zeros(nWin/2,1)];
            for n = 1:nFrames
                grain = x(pin+1:pin+nWin).*win;
                f = fft(grain,NFFT);
                Y(:,n) = f(1:nBins);
                T(n) = pin + 1;
                pin = pin + nHop;
            end
        end
        
        function y = istft(app,X,nHop,win)
            nWin = length(win);
            [nBins,nFrames] = size(X);
            NFFT = (nBins-1)*2;
            
            % Length of output
            L = (nFrames-1)*nHop + nWin;
            y = zeros(L,1);            
            
            % Normalization coefficient
            norm_coef = app.ola_norm_coef(win,nHop);
            
            % Compute two-sided spectrogram
            XF = zeros(NFFT,nFrames);
            XF(1:nBins,:) = X(1:nBins,:);
            XF(nBins+1:end,:) = conj(flipud(X(2:end-1,:)));
            
            % Overlap-add synthesis
            p = 0;
            for n = 1:nFrames
                grain = real(ifft(XF(:,n)));
                grain = grain(1:nWin).*win ./ norm_coef;
                y(p+1:p+nWin) = y(p+1:p+nWin) + grain;
                p = p + nHop;
            end
        end
        
        function y = ola_norm_coef(app,win,nHop)

            nWin = length(win);
            
            win = win .* win;
            idx = nWin / 2 + 1;
            y = win(idx);
            
            m = 1;
            i = idx - m * nHop;
            j = idx + m * nHop;
            while i > 0 &&  j <= nWin
                y = y + win(i) + win(j);
                m = m + 1;
                i = idx - m * nHop;
                j = idx + m * nHop;
            end
        end
        
        function [tri_fin,k,n] = findArtifacts(app,X)
            T0 = -110;
            tol_edge = 20;
            locMin = islocalmax(X,2);
            %locMin(locMin.*X < T0) = 0;
            locMin(locMin.*X > T0) = 0;
            % add check for max T0 energy value in low frequencies
            [k,n] = find(locMin==1);
            tri = delaunay([k n]);
            tri_fin = [];
            
            for i = 1:size(tri,1)
                %pos = [k(tri(i,:)) n(tri(i,:))]';
                %D = dist(pos);
                if ((abs(k(tri(i,1)) - k(tri(i,2))) > tol_edge)...
                        || (abs(k(tri(i,3)) - k(tri(i,2))) > tol_edge) ...
                        || (abs(k(tri(i,3)) - k(tri(i,1))) > tol_edge))
                %if ~isempty(D(D>tol_edge))
                        tri_fin = [tri_fin; tri(i,:)];
                end
%                 
%                 
%                 pos = [1+0*k(tri(i,:)) k(tri(i,:))];
%                 D = dist(pos);
%                 if isempty(D(D>tol_edge))
%                     tri_fin = [tri_fin; tri(i,:)];
%                 end
            end
        end   
            
        function loc = findlocalmax(app,X,T0)
            %T0 = -110;
            loc = islocalmax(X);
            %locMin(locMin.*X < T0) = 0;
            loc(loc.*X > T0) = 0;
            CC = bwconncomp(loc);

            numPixels = cellfun(@numel,CC.PixelIdxList);
            [biggest,idx] = max(numPixels);
            loc(CC.PixelIdxList{idx}) = 0;
            
        end
        
        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
        function separate(app)
            app.StatusLamp.Color = 'r';
            app.activeLabel.Text = "(Processing)";
            drawnow
            switch app.MethodDropDown.Value
                case "HPR"
                    app.HPR();
                case "HPR (Single Stage)"
                    app.HPR_single();
                case "StructureTensor"
                    app.HPR_st();
                case "HP (Soft Mask)"
                    app.HP();
                case "HP (Hard Mask)"
                    app.HPHard();
                case "Fuzzy"
                    app.Fuzzy();
            end
            
            app.StatusLamp.Color = 'g';
            app.activeLabel.Text = "(Active)";
            drawnow
                       
        end
        
        function Rt = transientness(~,X,nMedianH,nMedianV)
            X_v_median = medfilt1(abs(X),nMedianV,[],1);
            X_h_median = medfilt1(abs(X),nMedianH,[],2);
            Rt = X_v_median ./ (X_v_median + X_h_median + eps);
        end
        
        function [Mh,Mp,Mr] = structureTensor(app, X)
            [N,K] = size(X);
            S = 20 * log10(abs(X));
            
            Gx = [-47 0 47; -162 0 162; -47 0 47];  Gy = Gx';
            S_time = conv2(S,Gx,'same');    S_freq = conv2(S,Gy,'same');
    
            % Stucture tensor
            T11 = imgaussfilt(S_time .* S_time, app.params.sgm, 'FilterSize',9);
            T12 = imgaussfilt(S_time .* S_freq, app.params.sgm, 'FilterSize',9); T21 = T12;
            T22 = imgaussfilt(S_freq .* S_freq, app.params.sgm, 'FilterSize',9);
            
            R = zeros(N,K); C = R; Mh = R; Mp = R; Mr = R;
            
            for n = 1:N
                for k = 1:K
                    T = [T11(n,k) T12(n,k); T21(n,k) T22(n,k)];
                    T(isnan(T)) = -inf;
                    [V,D] = eig(T); D = diag(D);
                    alpha = atan(V(2,1)/V(1,1));
                                                            
                    % Anisotropy
                    if sum(D) >= app.params.tol
                        C(n,k) = ((D(2) - D(1))./(D(2) + D(1))).^2;
                    end
                    
                    % Frequency change rate
                    R(n,k) = tan(alpha) * app.Audio.fs.^2 / (app.nHop.r0 * app.nWin.r0);
                    
                    if C(n,k) > app.params.c
                        if abs(R(n,k)) <= app.params.r(1)
                            Mh(n,k) = 1;
                        elseif abs(R(n,k)) > app.params.r(2)
                            Mp(n,k) = 1;
                        end
                    end
                    
                    Mr(n,k) = 1 - Mh(n,k) - Mp(n,k);
                end
            end
            
        end
        
        function HPR(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r1,app.nHop.r1,...
                app.NFFT.r1);
            Rt = app.transientness(X,app.nMedianH.r1,app.nMedianV.r1);
            Rs = 1-Rt;
            S = (Rs./(Rt+eps)) > app.params.beta;
            T = (Rt./(Rs+eps)) >= app.params.beta;
            N = 1 - (S+T);
            
            
            app.Audio.tonal = app.istft(S.*X, app.nHop.r1, app.win.r1);
            x_res = app.istft((T+N).*X, app.nHop.r1, app.win.r1);
            %app.plotSTFT(S.*X, 'S');
            
            [X,~] = app.stft(x_res,app.win.r2,app.nHop.r2,...
                app.NFFT.r2);
                      
            Rt = app.transientness(X,app.nMedianH.r2,app.nMedianV.r2);
            Rs = 1-Rt;
            S = (Rs./(Rt+eps)) > app.params.beta;
            T = (Rt./(Rs+eps)) >= app.params.beta;
            N = 1 - (S+T);
            
            
            %app.plotSTFT(T.*X, 'T');
            %app.plotSTFT((S+N).*X, 'N');
            
            app.Audio.transient = app.istft(T.*X, app.nHop.r2, app.win.r2);
            app.Audio.noise = app.istft((S+N).*X, app.nHop.r2, app.win.r2);
            drawnow
        end

        function HPR_single(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r0,app.nHop.r0,...
                app.NFFT.r0);
            Rt = app.transientness(X,app.nMedianH.r0,app.nMedianV.r0);
            Rs = 1-Rt;
            S = (Rs./(Rt+eps)) > app.params.beta;
            T = (Rt./(Rs+eps)) >= app.params.beta;
            N = 1 - (S+T);
                        
            app.Audio.tonal = app.istft(S.*X, app.nHop.r0, app.win.r0); 
            app.Audio.transient = app.istft(T.*X, app.nHop.r0, app.win.r0);
            app.Audio.noise = app.istft(N.*X, app.nHop.r0, app.win.r0);
            drawnow
        end
        
        function HP(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r0,app.nHop.r0,...
                app.NFFT.r0);
            Rt = app.transientness(X,app.nMedianH.r0,app.nMedianV.r0);
            app.Audio.tonal = app.istft((1-Rt).*X, app.nHop.r0, app.win.r0);
            %app.plotSTFT((1-Rt).*X, 'S');
            %app.plotSTFT(Rt.*X, 'T');
            %app.plotSTFT(0.*X, 'N');
            
            app.Audio.transient = app.istft(Rt.*X, app.nHop.r0, app.win.r0);
            app.Audio.noise = zeros(size(app.Audio.transient));
            drawnow                       
        end
        
        function HPHard(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r0,app.nHop.r0,...
                app.NFFT.r0);
            Rt = app.transientness(X,app.nMedianH.r0,app.nMedianV.r0);
            Rs = 1-Rt;
            S = (Rs./(Rt+eps)) > 1;
            T = (Rt./(Rs+eps)) >= 1;
                        
            app.Audio.tonal = app.istft(S.*X, app.nHop.r0, app.win.r0); 
            app.Audio.transient = app.istft(T.*X, app.nHop.r0, app.win.r0);
            app.Audio.noise = zeros(size(app.Audio.transient));
            drawnow                       
        end
        
        function Fuzzy(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r0,app.nHop.r0,...
                app.NFFT.r0);
            Rt = app.transientness(X,app.nMedianH.r0,app.nMedianV.r0); 
            Rn = 1-sqrt(abs(1-2*Rt));
            app.Audio.tonal = app.istft((1-Rt-0.5*Rn).*X, app.nHop.r0, app.win.r0);            
            app.Audio.transient = app.istft((Rt-0.5*Rn).*X, app.nHop.r0, app.win.r0);
            app.Audio.noise = app.istft(Rn.*X, app.nHop.r0, app.win.r0);
            drawnow                       
        end
        
        function HPR_st(app)
            [X,~] = app.stft(app.Audio.inputdata,app.win.r0,app.nHop.r0,...
                app.NFFT.r0);
            [S,T,N] = app.structureTensor(X);
            app.Audio.tonal = app.istft(S.*X, app.nHop.r0, app.win.r0);
   
            app.Audio.transient = app.istft(T.*X, app.nHop.r0, app.win.r0);
            app.Audio.noise = app.istft(N.*X, app.nHop.r0, app.win.r0);
            
            drawnow
        end
        
        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
        function playAudio(app)
            cond = 1;
            while cond
                %if app.Player.CurrentSample == 1
                if app.Position == 1
                    play(app.Player);
                else
                    play(app.Player,app.Position);
                end
                while isplaying(app.Player)
                    app.Position = app.Player.CurrentSample;
                    app.lineDraw();
                    
                    drawnow
                end
                drawnow
                cond = app.LoopCheckBox.Value & app.STOPButton.Enable;
            end
            app.STOPButton.Enable = 0;
            app.STOPButton.Value = 0;
            app.PLAYButton.Enable = 1;
            drawnow;
        end
        
        function stopAudio(app)
            pause(app.Player)
            app.STOPButton.Enable = 0;
            app.STOPButton.Value = 0;
            app.PLAYButton.Enable = 1;
            drawnow
        end
        
        function updateAudio(app)
            ccc = 0;
            if isplaying(app.Player)               
                ccc = 1;
            end
            
            app.Audio.data = app.TonesSlider.Value / 100 * app.Audio.tonal + ...
                app.TransientsSlider.Value / 100 * app.Audio.transient + ...
                app.NoiseSlider.Value / 100 * app.Audio.noise;
            
                      
            app.Position = app.Player.CurrentSample;
            app.Player = audioplayer(app.Audio.data, app.Audio.fs);
            
            if ccc == 1
                play(app.Player,app.Position);
            else
                app.Position = 1;
            end
            
            % update position
            figure(app.UIFigure);
            cla(app.UIAxes2);
            time = 1:app.Audio.fs/2:length(app.Audio.data);
            plot(app.UIAxes2,app.Audio.data,'y','ButtonDownFcn',@app.lineCallback);
            axis(app.UIAxes2, 'tight'); hold(app.UIAxes2, 'on');
            xticks(app.UIAxes2,time);
            xticklabels(app.UIAxes2,string((time-1)/app.Audio.fs));
            plot(app.UIAxes2,[app.Position app.Position],[-1 1],'r')
            grid(app.UIAxes2, "on");
            drawnow
            app.plotSpectrum();
            drawnow
            
            app.TabGroup.SelectedTab = app.EvaluationTab;
            totalEnergy = sum(abs(app.Audio.data).^2);
            app.ESLabel.Text = string(100*sum(abs(app.TonesSlider.Value / 100 * app.Audio.tonal).^2)./totalEnergy); 
            app.ETLabel.Text = string(100*sum(abs(app.TransientsSlider.Value / 100 * app.Audio.transient).^2)./totalEnergy); 
            app.ENLabel.Text = string(100*sum(abs(app.NoiseSlider.Value / 100 * app.Audio.noise).^2)./totalEnergy); 
                     
            
        end
        
        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  
        
        function lineDraw(app)
            delete(app.UIAxes2.Children(1));
            if app.Position > length(app.Audio.tonal)
                app.Position = 1;
            end
            
            plot(app.UIAxes2,[app.Position app.Position],[-1 1],'r','LineWidth',2)
            app.plotSpectrum();
            app.gaugeUpdate();
            drawnow
        end
        
        function lineCallback(app,src,event)
            app.Position = event.IntersectionPoint(1);
            app.lineDraw();
        end
        
        function plotSpectrum(app)
            
            app.TabGroup.SelectedTab = app.SpectrumTab;
            if app.Position < app.FFTsize/2
                n = app.FFTsize/2 - app.Position;
                X = abs(fft([zeros(n,1); app.Audio.data(1:app.Position+app.FFTsize/2)],app.FFTsize)./app.FFTsize).^2; 
            elseif app.Position + app.FFTsize > app.Length
                n = app.Length - app.Position - app.FFTsize;
                X = abs(fft([app.Audio.data(app.Length-app.FFTsize/2+1:app.Length); zeros(n,1)],app.FFTsize)./app.FFTsize).^2; 
            else
                X = abs(fft(app.Audio.data(app.Position-app.FFTsize/2+1:app.Position+app.FFTsize/2),app.FFTsize)./app.FFTsize).^2;                
            end
            %cla(app.Spectrum)
            freq = 1:app.Audio.fs/app.FFTsize:app.Audio.fs/2;
            semilogx(app.Spectrum,freq,10*log10(X(1:end/2)),'y');
            %axis(app.Spectrum, "tight"); 
            xlim(app.Spectrum, [50 app.Audio.fs/2]);
            ylim(app.Spectrum, [-90 0])
            xticks(app.Spectrum,[50 100 200 500 1000 2000 5000 10000 20000]);
            %xticklabels(app.Spectrum,string([1 200 500 1000 2000 5000 10000 20000]));
            grid(app.Spectrum, "on");
            
        end
        
        function plotSTFTold(app)
%             [SS,f,t] = spectrogram(app.Audio.tonal,hann(app.FFTsize),app.FFTsize/2,app.FFTsize,app.Audio.fs,'onesided');
%             [t,f] = meshgrid(t,f);
%             ST = spectrogram(app.Audio.transient,hann(app.FFTsize),app.FFTsize/2,app.FFTsize,app.Audio.fs,'onesided');
%             SN = spectrogram(app.Audio.noise,hann(app.FFTsize),app.FFTsize/2,app.FFTsize,app.Audio.fs,'onesided');
            app.StatusLamp.Color = 'y';
            app.activeLabel.Text = "(Plotting)";
            drawnow
            
            NFFT = app.FFTsize;
            
            app.TabGroup.SelectedTab = app.TonalTab;
            drawnow
            [X,~] = app.stft(app.Audio.tonal/max(abs(app.Audio.inputdata)),hann(NFFT),NFFT/2,NFFT);
            X = 20*log10(abs(X)./NFFT);
            %X(X < -130) = -130;
            X(X < -91) = -91;
            t = linspace(0,(length(app.Audio.tonal)-1)/app.Audio.fs,size(X,2));
            f = linspace(1,app.Audio.fs/2,size(X,1));
            [t2,f2] = meshgrid(t,f);
            hold(app.STFT_S,"off")
            %mesh(app.STFT_S,t,f,20*log10(abs(X))); 
            mesh(app.STFT_S,t,f,X); 
            view(app.STFT_S,0,90); 
            grid(app.STFT_S,"on")
            xlim(app.STFT_S,[t(1,1) t(1,end)]); ylim(app.STFT_S,[f(1,1) f(1,end)]);
            zlim(app.STFT_S, [-90 0]); 
            hold(app.STFT_S, "on")
            
            loc = app.findlocalmax(X,app.Threshold.S);
            app.Audio.artifacts.S = loc;
            
            loc = 20*log10(loc+eps);
            plot3(app.STFT_S,t2,f2,loc,'r*');
            %[tri,k,n] = app.findArtifacts(X);
            %triplot(tri,t(n),f(k),'r','Parent',app.STFT_S);
            colorbar(app.STFT_S,'Color',[1 1 1],'Limits',[-90 0]);
            %colormap(app.STFT_S,"jet")
            colormap(app.STFT_S,"parula")
            drawnow
            app.TabGroup.SelectedTab = app.TransientTab;
            drawnow
            [X,~] = app.stft(app.Audio.transient/max(abs(app.Audio.inputdata)),hann(NFFT),NFFT/2,NFFT);
            X = 20*log10(abs(X)./NFFT);
            %X(X < -130) = -130;
            X(X < -91) = -91; 
            hold(app.STFT_T,"off")
            %mesh(app.STFT_T,t,f,20*log10(abs(X))); 
            mesh(app.STFT_T,t,f,X);
            view(app.STFT_T,0,90); 
            grid(app.STFT_T,"on")
            xlim(app.STFT_T,[t(1,1) t(1,end)]); ylim(app.STFT_T,[f(1,1) f(1,end)]);
            %zlim(app.STFT_T, [-120 0])
            zlim(app.STFT_T, [-90 0])
            hold(app.STFT_T, "on")
            loc = app.findlocalmax(X,app.Threshold.T);
            
            app.Audio.artifacts.T = loc;
            
            loc = 20*log10(loc+eps);
            plot3(app.STFT_T,t2,f2,loc,'r*');
            %[tri,k,n] = app.findArtifacts(X);
            %triplot(tri,t(n),f(k),'r','Parent',app.STFT_T);
            colorbar(app.STFT_T,'Color',[1 1 1],'Limits',[-90 0]);
            colormap(app.STFT_T,"parula")
            drawnow
            app.TabGroup.SelectedTab = app.NoiseTab;
            drawnow
            [X,~] = app.stft(app.Audio.noise/max(abs(app.Audio.inputdata)),hann(NFFT),NFFT/2,NFFT);
            X = 20*log10(abs(X)./NFFT);
            %X(X < -130) = -130; 
            X(X < -91) = -91;
            hold(app.STFT_N,"off")
            %mesh(app.STFT_N,t,f,20*log10(abs(X)));
            mesh(app.STFT_N,t,f,X);
            view(app.STFT_N,0,90); 
            grid(app.STFT_N,"on")
            xlim(app.STFT_N,[t(1,1) t(1,end)]); ylim(app.STFT_N,[f(1,1) f(1,end)]);
            %zlim(app.STFT_N, [-120 0])
            zlim(app.STFT_N, [-90 0])
            colorbar(app.STFT_N,'Color',[1 1 1],'Limits',[-90 0]); 
            colormap(app.STFT_N,"parula")
            %colormap(app.STFT_N,"jet")
          
            drawnow
            
            app.TabGroup.SelectedTab = app.EvaluationTab;
            N = size(app.Audio.artifacts.S,2);
            K = size(app.Audio.artifacts.S,1);
            app.ASLabel.Text = string(100*sum(app.Audio.artifacts.S(:))/N/K);
            app.ATLabel.Text = string(100*sum(app.Audio.artifacts.T(:))/N/K);
            app.ANLabel.Text = "0";
            
            hold(app.ArtifactsAxes,"off")
            plot(app.ArtifactsAxes,t,100*sum(app.Audio.artifacts.S,1)/K);
            hold(app.ArtifactsAxes,"on")
            plot(app.ArtifactsAxes,t,100*sum(app.Audio.artifacts.T,1)/K);
            legend(app.ArtifactsAxes,{'S','T'},'TextColor',[1 1 1])
            
            drawnow
            app.TabGroup.SelectedTab = app.SpectrumTab;
            
            app.StatusLamp.Color = 'g';
            app.activeLabel.Text = "(Active)";
            drawnow
            
        end
        
        function plotSTFT(app,X,fset)
            t = linspace(0,(length(app.Audio.tonal)-1)/app.Audio.fs,size(X,2));
            f = linspace(1,app.Audio.fs/2,size(X,1));
            [t,f] = meshgrid(t,f);
        switch fset
            case "S"
                app.TabGroup.SelectedTab = app.TonalTab;
                mesh(app.STFT_S,t,f,20*log10(abs(X))); view(app.STFT_S,0,90); grid(app.STFT_S,"on")
                xlim(app.STFT_S,[t(1,1) t(1,end)]); ylim(app.STFT_S,[f(1,1) f(end,1)]);
            case "T"
                app.TabGroup.SelectedTab = app.TransientTab;
                mesh(app.STFT_T,t,f,20*log10(abs(X))); view(app.STFT_T,0,90); grid(app.STFT_T,"on")
                xlim(app.STFT_T,[t(1,1) t(1,end)]); ylim(app.STFT_T,[f(1,1) f(end,1)]);
            case "N"  
                app.TabGroup.SelectedTab = app.NoiseTab;
                mesh(app.STFT_N,t,f,20*log10(abs(X))); view(app.STFT_N,0,90); grid(app.STFT_N,"on")
                xlim(app.STFT_N,[t(1,1) t(1,end)]); ylim(app.STFT_N,[f(1,1) f(end,1)]);
        end
             app.TabGroup.SelectedTab = app.SpectrumTab;
        end
        
        function gaugeUpdate(app)
            if app.Position == 1
               app.GaugeS.Value = 0; app.GaugeT.Value = 0; app.GaugeN.Value = 0;
            else
            
            ts = app.Position - app.nS; ts(ts<1)=1;
            tt = app.Position - app.nT; tt(tt<1)=1;
            tn = app.Position - app.nN; tn(tn<1)=1;
            
%             enS = mean(gaugeScaling(app,(app.TonesSlider.Value/100 * app.Audio.tonal(ts:app.Position)).^2)) + eps;
%             enT = mean(gaugeScaling(app,(app.TransientsSlider.Value/100 * app.Audio.transient(tt:app.Position)).^2))+ eps;
%             enN = mean(gaugeScaling(app,(app.NoiseSlider.Value/100 * app.Audio.noise(tn:app.Position)).^2))+ eps;   
%             
%             app.GaugeS.Value = 100*enS/(enS+enT+enN); 
%             app.GaugeT.Value = 100*enT/(enS+enT+enN); 
%             app.GaugeN.Value = 100*enN/(enS+enT+enN); 
            app.GaugeS.Value = floor(mean(gaugeScaling(app,(app.TonesSlider.Value/100 * app.Audio.tonal(ts:app.Position)).^2)));
            app.GaugeT.Value = floor(mean(gaugeScaling(app,(app.TransientsSlider.Value/100 * app.Audio.transient(tt:app.Position)).^2)));
            app.GaugeN.Value = floor(mean(gaugeScaling(app,(app.NoiseSlider.Value/100 * app.Audio.noise(tn:app.Position)).^2))); 
            
            end
        end
        
        function y = gaugeScaling(~,x)
            y = (60+10*log10((x).^2))*100/60;
            %y = (60+10*log10((x).^2));
            y(y<0) = 0; y(isnan(y)) = 0;
            
        end

        %% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
        function readInput(app)
            [file,path] = uigetfile({'*.wav';'*.mp3'});
            [x,app.Audio.fs] = audioread([path,file]);
            app.Audio.inputdata = x(:,1)./max(abs(x(:,1)));
            app.Player = audioplayer(app.Audio.inputdata, app.Audio.fs);
            app.Position = 1; 
            
            figure(app.UIFigure);
            app.GaugeS.Value = 0; app.GaugeT.Value = 0; app.GaugeN.Value = 0;
            
            time = 1:app.Audio.fs/2:length(app.Audio.inputdata);
            plot(app.UIAxes2,app.Audio.inputdata,'y','ButtonDownFcn',@app.lineCallback);
            axis(app.UIAxes2, 'tight'); hold(app.UIAxes2, 'on');
            xticks(app.UIAxes2,time);
            xticklabels(app.UIAxes2,string((time-1)/app.Audio.fs));
            
            plot(app.UIAxes2,[1 1],[-1 1],'r','LineWidth',2)
            grid(app.UIAxes2,"on")
            drawnow
            
            
            app.resetInitialValues();
            app.separate();
            
            app.updateAudio();
            app.Length = length(app.Audio.tonal);
            
            drawnow
            app.plotSTFTold();
            drawnow
            
            
            
        end
        
        function saveOutput(app)
            [file, path] = uiputfile('*.wav','Save audio mix as:');
            audiowrite([path,file],app.Audio.data,app.Audio.fs);
        end
        
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
%             fPos = app.UIFigure.Position;
%             sPos = get(0,ScreenSize');
%             if sPos(3) < fPos(3)-99
%                 ratio = (fPos(3)-99)/sPos(3);
%                 app.UIFigure.Position = app.UIFigure.Position -99;
%                 app.UIFigure.Position(3) = app.UIFigure.Position(3)/(1.5*ratio);
%                 app.UIFigure.Position(4) = app.UIFigure.Position(4)/(1.5*ratio);
%             end
            %pp.FigurePosition = (fPos(3)-fPos(1))/(fPos(4)-fPos(2));
            app.UIFigure.Name = 'SiTraNo v1.0';
            %app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @myUpdateAppLayout, true);
            app.readInput();
        end

        % Callback function: OpenButton, OpenMenu
        function OpenButtonPushed(app, event)
            app.readInput();
        end

        % Value changed function: TonesCheckBox
        function TonesCheckBoxValueChanged(app, event)
            if app.TonesCheckBox.Value == 1
                app.TonesSlider.Value = 100;               
            else
                app.TonesSlider.Value = 0;
            end
            app.updateAudio();
        end

        % Value changed function: TransientsCheckBox
        function TransientsCheckBoxValueChanged(app, event)
            if app.TransientsCheckBox.Value == 1
                app.TransientsSlider.Value = 100;               
            else
                app.TransientsSlider.Value = 0;
            end
            app.updateAudio();
        end

        % Value changed function: NoiseCheckBox
        function NoiseCheckBoxValueChanged(app, event)
            if app.NoiseCheckBox.Value == 1
                app.NoiseSlider.Value = 100;               
            else
                app.NoiseSlider.Value = 0;
            end
            app.updateAudio();
        end

        % Callback function
        function SaveButtonPushed(app, event)
            app.saveOutput(app);           
        end

        % Value changed function: TonesSlider
        function TonesSliderValueChanged(app, event)
            app.updateAudio();
        end

        % Value changed function: TransientsSlider
        function TransientsSliderValueChanged(app, event)
            app.updateAudio();
        end

        % Value changed function: NoiseSlider
        function NoiseSliderValueChanged(app, event)
            app.updateAudio();
        end

        % Callback function
        function PLAYButtonPushed(app, event)
            app.PLAYButton.Enable = 0;
            app.PLAYButton.Value = 0;
            app.STOPButton.Enable = 1;
            app.playAudio();
            drawnow;
        end

        % Callback function
        function SaveButtonPushed2(app, event)
            app.saveOutput(app); 
        end

        % Value changed function: STOPButton
        function STOPButtonValueChanged(app, event)
            app.STOPButton.Enable = 0;
            app.STOPButton.Value = 0;
            app.PLAYButton.Enable = 1;
            app.stopAudio();
            drawnow;
        end

        % Value changed function: PLAYButton
        function PLAYButtonValueChanged(app, event)
            app.PLAYButton.Enable = 0;
            app.PLAYButton.Value = 0;
            app.STOPButton.Enable = 1;
            app.playAudio();
            drawnow;
        end

        % Value changed function: MethodDropDown
        function MethodDropDownValueChanged(app, event)
            ccc = 0;
            if isplaying(app.Player)               
                ccc = 1;
                pause(app.Player);
                position = app.Position;
            end
            
            app.separate();
            app.updateAudio();
            app.plotSTFTold();
            drawnow
            
            if ccc == 1
                play(app.Player,position);
            end
            drawnow
        end

        % Button down function: UIAxes2
        function UIAxes2ButtonDown(app, event)
            app.Position = event.IntersectionPoint(1);
            app.lineDraw();
            drawnow;
        end

        % Value changed function: NFFTEditField
        function NFFTEditFieldValueChanged(app, event)
            app.FFTsize = app.NFFTEditField.Value;
            app.plotSpectrum();
            drawnow;
        end

        % Value changed function: ThresholdTEditField
        function ThresholdTEditFieldValueChanged(app, event)
            app.Threshold.T = app.ThresholdTEditField.Value;
            
        end

        % Value changed function: ThresholdSEditField
        function ThresholdSEditFieldValueChanged(app, event)
            app.Threshold.S = app.ThresholdSEditField.Value;
            
        end

        % Button pushed function: UpdatePlotsButton
        function UpdatePlotsButtonPushed(app, event)
            app.plotSTFTold();
            drawnow;
        end

        % Callback function: SaveButton, SaveMenu
        function SaveButtonPushed3(app, event)
            app.saveOutput();
        end

        % Menu selected function: VersionLogMenu
        function VersionLogMenuSelected(app, event)
            %fig = uifigure;
            message = sprintf(['SeparaTion - v1.0.0 \n ... ' ...
                'Log: updated Save Button, HP Hard Mask method, Menu buttons.']);
            uialert(app.UIFigure,message,'Version Log','Icon','info');
        end

        % Menu selected function: AboutMenu
        function AboutMenuSelected(app, event)
            %fig = uifigure;
            message = sprintf(['SeparaTioN is an app for decomposition of sounds into tonal, transient and noise component. \n' ...
                'It has been developed by Leonardo Fierro, Doctoral Student, Acoustics Lab, Aalto University in 2021 for the complementary DaFx paper.']);
            uialert(app.UIFigure,message,'About SeparaTion... ','Icon','');
        end

        % Menu selected function: GlobalparametersMenu
        function GlobalparametersMenuSelected(app, event)
            message = sprintf(['This feature is not available yet, but will be implemented in the next update. Please check on the GitHub Page (https://himynameisfuego.github.io/SeparaTioN/) if a more recent version of SeparaTioN is available.']);
            uialert(app.UIFigure,message,'Set global parameters','Icon','Info');
        end

        % Menu selected function: WebsiteMenu
        function WebsiteMenuSelected(app, event)
            web('https://himynameisfuego.github.io/SeparaTioN/', '-browser')
        end

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, event)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {781, 781};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {880, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 1259 785];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.Resize = 'off';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {880, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.ForegroundColor = [1 1 1];
            app.LeftPanel.BackgroundColor = [0.149 0.149 0.149];
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create GaugeN
            app.GaugeN = uigauge(app.LeftPanel, 'semicircular');
            app.GaugeN.BackgroundColor = [0.149 0.149 0.149];
            app.GaugeN.FontColor = [1 1 1];
            app.GaugeN.Position = [40 36 120 65];

            % Create TabGroup
            app.TabGroup = uitabgroup(app.LeftPanel);
            app.TabGroup.Position = [6 256 868 519];

            % Create SpectrumTab
            app.SpectrumTab = uitab(app.TabGroup);
            app.SpectrumTab.Title = 'Spectrum';
            app.SpectrumTab.BackgroundColor = [0.149 0.149 0.149];
            app.SpectrumTab.ForegroundColor = [0.149 0.149 0.149];

            % Create SpectrumAnalyzerLabel
            app.SpectrumAnalyzerLabel = uilabel(app.SpectrumTab);
            app.SpectrumAnalyzerLabel.HorizontalAlignment = 'center';
            app.SpectrumAnalyzerLabel.FontWeight = 'bold';
            app.SpectrumAnalyzerLabel.FontColor = [1 1 1];
            app.SpectrumAnalyzerLabel.Position = [330 456 208 40];
            app.SpectrumAnalyzerLabel.Text = 'Spectrum Analyzer';

            % Create Spectrum
            app.Spectrum = uiaxes(app.SpectrumTab);
            xlabel(app.Spectrum, 'Frequency (Hz)')
            ylabel(app.Spectrum, 'Magnitude (dB)')
            app.Spectrum.XColor = [1 1 1];
            app.Spectrum.YColor = [1 1 1];
            app.Spectrum.ZColor = [1 1 1];
            app.Spectrum.Color = [0.149 0.149 0.149];
            app.Spectrum.GridColor = [1 1 1];
            app.Spectrum.Position = [11 16 846 462];

            % Create TonalTab
            app.TonalTab = uitab(app.TabGroup);
            app.TonalTab.Title = 'Tonal';
            app.TonalTab.BackgroundColor = [0.149 0.149 0.149];

            % Create STFTTonalComponentLabel
            app.STFTTonalComponentLabel = uilabel(app.TonalTab);
            app.STFTTonalComponentLabel.HorizontalAlignment = 'center';
            app.STFTTonalComponentLabel.FontWeight = 'bold';
            app.STFTTonalComponentLabel.FontColor = [1 1 1];
            app.STFTTonalComponentLabel.Position = [330 456 208 40];
            app.STFTTonalComponentLabel.Text = 'STFT - Tonal Component';

            % Create STFT_S
            app.STFT_S = uiaxes(app.TonalTab);
            xlabel(app.STFT_S, 'Time (s)')
            ylabel(app.STFT_S, 'Frequency (Hz)')
            zlabel(app.STFT_S, 'Magnitude (dB)')
            app.STFT_S.XColor = [1 1 1];
            app.STFT_S.YColor = [1 1 1];
            app.STFT_S.ZColor = [1 1 1];
            app.STFT_S.Color = [0.149 0.149 0.149];
            app.STFT_S.GridColor = [1 1 1];
            app.STFT_S.Position = [11 16 846 462];

            % Create TransientTab
            app.TransientTab = uitab(app.TabGroup);
            app.TransientTab.Title = 'Transient';
            app.TransientTab.BackgroundColor = [0.149 0.149 0.149];

            % Create STFTTransientComponentLabel
            app.STFTTransientComponentLabel = uilabel(app.TransientTab);
            app.STFTTransientComponentLabel.HorizontalAlignment = 'center';
            app.STFTTransientComponentLabel.FontWeight = 'bold';
            app.STFTTransientComponentLabel.FontColor = [1 1 1];
            app.STFTTransientComponentLabel.Position = [330 456 208 40];
            app.STFTTransientComponentLabel.Text = 'STFT - Transient Component';

            % Create STFT_T
            app.STFT_T = uiaxes(app.TransientTab);
            xlabel(app.STFT_T, 'Time (s)')
            ylabel(app.STFT_T, 'Frequency (Hz)')
            zlabel(app.STFT_T, 'Magnitude (dB)')
            app.STFT_T.XColor = [1 1 1];
            app.STFT_T.YColor = [1 1 1];
            app.STFT_T.ZColor = [1 1 1];
            app.STFT_T.Color = [0.149 0.149 0.149];
            app.STFT_T.GridColor = [1 1 1];
            app.STFT_T.Position = [11 16 846 462];

            % Create NoiseTab
            app.NoiseTab = uitab(app.TabGroup);
            app.NoiseTab.Title = 'Noise';
            app.NoiseTab.BackgroundColor = [0.149 0.149 0.149];

            % Create STFTNoiseComponentLabel
            app.STFTNoiseComponentLabel = uilabel(app.NoiseTab);
            app.STFTNoiseComponentLabel.HorizontalAlignment = 'center';
            app.STFTNoiseComponentLabel.FontWeight = 'bold';
            app.STFTNoiseComponentLabel.FontColor = [1 1 1];
            app.STFTNoiseComponentLabel.Position = [330 456 208 40];
            app.STFTNoiseComponentLabel.Text = 'STFT - Noise Component';

            % Create STFT_N
            app.STFT_N = uiaxes(app.NoiseTab);
            xlabel(app.STFT_N, 'Time (s)')
            ylabel(app.STFT_N, 'Frequency (Hz)')
            zlabel(app.STFT_N, 'Magnitude (dB)')
            app.STFT_N.XColor = [1 1 1];
            app.STFT_N.YColor = [1 1 1];
            app.STFT_N.ZColor = [1 1 1];
            app.STFT_N.Color = [0.149 0.149 0.149];
            app.STFT_N.GridColor = [1 1 1];
            app.STFT_N.Position = [11 16 846 462];

            % Create EvaluationTab
            app.EvaluationTab = uitab(app.TabGroup);
            app.EvaluationTab.Title = 'Evaluation';
            app.EvaluationTab.BackgroundColor = [0.149 0.149 0.149];

            % Create ESLabel
            app.ESLabel = uilabel(app.EvaluationTab);
            app.ESLabel.HorizontalAlignment = 'center';
            app.ESLabel.FontWeight = 'bold';
            app.ESLabel.FontColor = [1 1 1];
            app.ESLabel.Position = [280 396 208 40];
            app.ESLabel.Text = 'ES';

            % Create BinswArtifactsLabel
            app.BinswArtifactsLabel = uilabel(app.EvaluationTab);
            app.BinswArtifactsLabel.HorizontalAlignment = 'center';
            app.BinswArtifactsLabel.FontWeight = 'bold';
            app.BinswArtifactsLabel.FontColor = [1 1 1];
            app.BinswArtifactsLabel.Position = [50 344 208 40];
            app.BinswArtifactsLabel.Text = 'Bins w/ Artifacts (%)';

            % Create STonalLabel
            app.STonalLabel = uilabel(app.EvaluationTab);
            app.STonalLabel.HorizontalAlignment = 'center';
            app.STonalLabel.FontWeight = 'bold';
            app.STonalLabel.FontColor = [1 1 1];
            app.STonalLabel.Position = [280 438 208 40];
            app.STonalLabel.Text = 'S (Tonal)';

            % Create TTransientLabel
            app.TTransientLabel = uilabel(app.EvaluationTab);
            app.TTransientLabel.HorizontalAlignment = 'center';
            app.TTransientLabel.FontWeight = 'bold';
            app.TTransientLabel.FontColor = [1 1 1];
            app.TTransientLabel.Position = [445 438 208 40];
            app.TTransientLabel.Text = 'T (Transient)';

            % Create NNoiseLabel
            app.NNoiseLabel = uilabel(app.EvaluationTab);
            app.NNoiseLabel.HorizontalAlignment = 'center';
            app.NNoiseLabel.FontWeight = 'bold';
            app.NNoiseLabel.FontColor = [1 1 1];
            app.NNoiseLabel.Position = [615 438 208 40];
            app.NNoiseLabel.Text = 'N (Noise)';

            % Create EnergyLabel_2
            app.EnergyLabel_2 = uilabel(app.EvaluationTab);
            app.EnergyLabel_2.HorizontalAlignment = 'center';
            app.EnergyLabel_2.FontWeight = 'bold';
            app.EnergyLabel_2.FontColor = [1 1 1];
            app.EnergyLabel_2.Position = [50 396 208 40];
            app.EnergyLabel_2.Text = 'Energy (%)';

            % Create ASLabel
            app.ASLabel = uilabel(app.EvaluationTab);
            app.ASLabel.HorizontalAlignment = 'center';
            app.ASLabel.FontWeight = 'bold';
            app.ASLabel.FontColor = [1 1 1];
            app.ASLabel.Position = [280 344 208 40];
            app.ASLabel.Text = 'AS';

            % Create ETLabel
            app.ETLabel = uilabel(app.EvaluationTab);
            app.ETLabel.HorizontalAlignment = 'center';
            app.ETLabel.FontWeight = 'bold';
            app.ETLabel.FontColor = [1 1 1];
            app.ETLabel.Position = [445 396 208 40];
            app.ETLabel.Text = 'ET';

            % Create ATLabel
            app.ATLabel = uilabel(app.EvaluationTab);
            app.ATLabel.HorizontalAlignment = 'center';
            app.ATLabel.FontWeight = 'bold';
            app.ATLabel.FontColor = [1 1 1];
            app.ATLabel.Position = [445 344 208 40];
            app.ATLabel.Text = 'AT';

            % Create ENLabel
            app.ENLabel = uilabel(app.EvaluationTab);
            app.ENLabel.HorizontalAlignment = 'center';
            app.ENLabel.FontWeight = 'bold';
            app.ENLabel.FontColor = [1 1 1];
            app.ENLabel.Position = [615 396 208 40];
            app.ENLabel.Text = 'EN';

            % Create ANLabel
            app.ANLabel = uilabel(app.EvaluationTab);
            app.ANLabel.HorizontalAlignment = 'center';
            app.ANLabel.FontWeight = 'bold';
            app.ANLabel.FontColor = [1 1 1];
            app.ANLabel.Position = [615 344 208 40];
            app.ANLabel.Text = 'AN';

            % Create ArtifactsperTimeFrameLabel
            app.ArtifactsperTimeFrameLabel = uilabel(app.EvaluationTab);
            app.ArtifactsperTimeFrameLabel.HorizontalAlignment = 'center';
            app.ArtifactsperTimeFrameLabel.FontWeight = 'bold';
            app.ArtifactsperTimeFrameLabel.FontColor = [1 1 1];
            app.ArtifactsperTimeFrameLabel.Position = [333 297 208 40];
            app.ArtifactsperTimeFrameLabel.Text = 'Artifacts per Time Frame';

            % Create ArtifactsAxes
            app.ArtifactsAxes = uiaxes(app.EvaluationTab);
            xlabel(app.ArtifactsAxes, 'Time (s)')
            ylabel(app.ArtifactsAxes, 'Artifacts (%)')
            app.ArtifactsAxes.Toolbar.Visible = 'off';
            app.ArtifactsAxes.XColor = [1 1 1];
            app.ArtifactsAxes.YColor = [1 1 1];
            app.ArtifactsAxes.ZColor = [1 1 1];
            app.ArtifactsAxes.Color = [0.149 0.149 0.149];
            app.ArtifactsAxes.GridColor = [1 1 1];
            app.ArtifactsAxes.MinorGridColor = [0.149 0.149 0.149];
            app.ArtifactsAxes.Position = [41 16 792 315];

            % Create GaugeT
            app.GaugeT = uigauge(app.LeftPanel, 'semicircular');
            app.GaugeT.BackgroundColor = [0.149 0.149 0.149];
            app.GaugeT.FontColor = [1 1 1];
            app.GaugeT.Position = [40 110 120 65];

            % Create GaugeS
            app.GaugeS = uigauge(app.LeftPanel, 'semicircular');
            app.GaugeS.BackgroundColor = [0.149 0.149 0.149];
            app.GaugeS.FontColor = [1 1 1];
            app.GaugeS.Position = [40 181 120 65];

            % Create TLabel
            app.TLabel = uilabel(app.LeftPanel);
            app.TLabel.HorizontalAlignment = 'center';
            app.TLabel.FontWeight = 'bold';
            app.TLabel.FontColor = [1 1 1];
            app.TLabel.Position = [84 110 33 39];
            app.TLabel.Text = 'T';

            % Create SLabel
            app.SLabel = uilabel(app.LeftPanel);
            app.SLabel.HorizontalAlignment = 'center';
            app.SLabel.FontWeight = 'bold';
            app.SLabel.FontColor = [1 1 1];
            app.SLabel.Position = [85 181 33 39];
            app.SLabel.Text = 'S';

            % Create NLabel
            app.NLabel = uilabel(app.LeftPanel);
            app.NLabel.HorizontalAlignment = 'center';
            app.NLabel.FontWeight = 'bold';
            app.NLabel.FontColor = [1 1 1];
            app.NLabel.Position = [84 39 33 39];
            app.NLabel.Text = 'N';

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.LeftPanel);
            xlabel(app.UIAxes2, 'Time (s)')
            app.UIAxes2.XColor = [1 1 1];
            app.UIAxes2.YColor = [1 1 1];
            app.UIAxes2.ZColor = [1 1 1];
            app.UIAxes2.Color = [0.149 0.149 0.149];
            app.UIAxes2.GridColor = [1 1 1];
            app.UIAxes2.ButtonDownFcn = createCallbackFcn(app, @UIAxes2ButtonDown, true);
            app.UIAxes2.Position = [173 27 686 218];

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.ForegroundColor = [0 0.4471 0.7412];
            app.RightPanel.BackgroundColor = [0.149 0.149 0.149];
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create activeLabel
            app.activeLabel = uilabel(app.RightPanel);
            app.activeLabel.FontColor = [1 1 1];
            app.activeLabel.Position = [195 646 100 22];
            app.activeLabel.Text = '(active)';

            % Create ParametersLabel
            app.ParametersLabel = uilabel(app.RightPanel);
            app.ParametersLabel.HorizontalAlignment = 'center';
            app.ParametersLabel.FontWeight = 'bold';
            app.ParametersLabel.FontColor = [1 1 1];
            app.ParametersLabel.Position = [116 489 208 40];
            app.ParametersLabel.Text = 'Parameters';

            % Create STOPButton
            app.STOPButton = uibutton(app.RightPanel, 'state');
            app.STOPButton.ValueChangedFcn = createCallbackFcn(app, @STOPButtonValueChanged, true);
            app.STOPButton.Text = 'STOP';
            app.STOPButton.Position = [37 39 100 22];

            % Create PLAYButton
            app.PLAYButton = uibutton(app.RightPanel, 'state');
            app.PLAYButton.ValueChangedFcn = createCallbackFcn(app, @PLAYButtonValueChanged, true);
            app.PLAYButton.Text = 'PLAY';
            app.PLAYButton.Position = [38 71 97 22];

            % Create SiTraNoLabel
            app.SiTraNoLabel = uilabel(app.RightPanel);
            app.SiTraNoLabel.BackgroundColor = [1 1 0.0667];
            app.SiTraNoLabel.HorizontalAlignment = 'center';
            app.SiTraNoLabel.FontName = 'Ubuntu';
            app.SiTraNoLabel.FontSize = 36;
            app.SiTraNoLabel.FontWeight = 'bold';
            app.SiTraNoLabel.Position = [46 695 291 75];
            app.SiTraNoLabel.Text = 'SiTraNo      ';

            % Create TonesCheckBox
            app.TonesCheckBox = uicheckbox(app.RightPanel);
            app.TonesCheckBox.ValueChangedFcn = createCallbackFcn(app, @TonesCheckBoxValueChanged, true);
            app.TonesCheckBox.Text = 'Tones';
            app.TonesCheckBox.FontWeight = 'bold';
            app.TonesCheckBox.FontColor = [1 1 1];
            app.TonesCheckBox.Position = [62 346 96 36];
            app.TonesCheckBox.Value = true;

            % Create LoopCheckBox
            app.LoopCheckBox = uicheckbox(app.RightPanel);
            app.LoopCheckBox.Text = 'Loop';
            app.LoopCheckBox.FontColor = [1 1 1];
            app.LoopCheckBox.Position = [162 64 96 36];

            % Create NoiseCheckBox
            app.NoiseCheckBox = uicheckbox(app.RightPanel);
            app.NoiseCheckBox.ValueChangedFcn = createCallbackFcn(app, @NoiseCheckBoxValueChanged, true);
            app.NoiseCheckBox.Text = 'Noise';
            app.NoiseCheckBox.FontWeight = 'bold';
            app.NoiseCheckBox.FontColor = [1 1 1];
            app.NoiseCheckBox.Position = [168 346 96 36];
            app.NoiseCheckBox.Value = true;

            % Create TransientsCheckBox
            app.TransientsCheckBox = uicheckbox(app.RightPanel);
            app.TransientsCheckBox.ValueChangedFcn = createCallbackFcn(app, @TransientsCheckBoxValueChanged, true);
            app.TransientsCheckBox.Text = 'Transients';
            app.TransientsCheckBox.FontWeight = 'bold';
            app.TransientsCheckBox.FontColor = [1 1 1];
            app.TransientsCheckBox.Position = [265 346 96 36];
            app.TransientsCheckBox.Value = true;

            % Create OpenButton
            app.OpenButton = uibutton(app.RightPanel, 'push');
            app.OpenButton.ButtonPushedFcn = createCallbackFcn(app, @OpenButtonPushed, true);
            app.OpenButton.Position = [240 70 97 24];
            app.OpenButton.Text = 'Open...';

            % Create TransientsSlider
            app.TransientsSlider = uislider(app.RightPanel);
            app.TransientsSlider.MajorTicks = [0 20 40 60 80 100];
            app.TransientsSlider.Orientation = 'vertical';
            app.TransientsSlider.ValueChangedFcn = createCallbackFcn(app, @TransientsSliderValueChanged, true);
            app.TransientsSlider.FontColor = [1 1 1];
            app.TransientsSlider.Position = [287 144 7 185];

            % Create Label_2
            app.Label_2 = uilabel(app.RightPanel);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.FontColor = [1 1 1];
            app.Label_2.Position = [241 135 25 22];
            app.Label_2.Text = '%';

            % Create NoiseSlider
            app.NoiseSlider = uislider(app.RightPanel);
            app.NoiseSlider.MajorTicks = [0 20 40 60 80 100];
            app.NoiseSlider.MajorTickLabels = {'0', '20', '40', '60', '80', '100'};
            app.NoiseSlider.Orientation = 'vertical';
            app.NoiseSlider.ValueChangedFcn = createCallbackFcn(app, @NoiseSliderValueChanged, true);
            app.NoiseSlider.FontColor = [1 1 1];
            app.NoiseSlider.Position = [190 144 7 185];

            % Create Label
            app.Label = uilabel(app.RightPanel);
            app.Label.HorizontalAlignment = 'right';
            app.Label.FontColor = [1 1 1];
            app.Label.Position = [144 135 25 22];
            app.Label.Text = '%';

            % Create TonesSlider
            app.TonesSlider = uislider(app.RightPanel);
            app.TonesSlider.MajorTicks = [0 20 40 60 80 100];
            app.TonesSlider.Orientation = 'vertical';
            app.TonesSlider.ValueChangedFcn = createCallbackFcn(app, @TonesSliderValueChanged, true);
            app.TonesSlider.FontColor = [1 1 1];
            app.TonesSlider.Position = [84 144 7 185];

            % Create Label_3
            app.Label_3 = uilabel(app.RightPanel);
            app.Label_3.HorizontalAlignment = 'right';
            app.Label_3.FontColor = [1 1 1];
            app.Label_3.Position = [38 135 25 22];
            app.Label_3.Text = '%';

            % Create MethodDropDown
            app.MethodDropDown = uidropdown(app.RightPanel);
            app.MethodDropDown.Items = {'HP (Hard Mask)', 'HP (Soft Mask)', 'HPR (Single Stage)', 'HPR', 'StructureTensor', 'Fuzzy'};
            app.MethodDropDown.ValueChangedFcn = createCallbackFcn(app, @MethodDropDownValueChanged, true);
            app.MethodDropDown.FontColor = [1 1 1];
            app.MethodDropDown.BackgroundColor = [0.149 0.149 0.149];
            app.MethodDropDown.Position = [133 558 200 35];
            app.MethodDropDown.Value = 'HPR';

            % Create MethodDropDownLabel
            app.MethodDropDownLabel = uilabel(app.RightPanel);
            app.MethodDropDownLabel.BackgroundColor = [0.149 0.149 0.149];
            app.MethodDropDownLabel.HorizontalAlignment = 'right';
            app.MethodDropDownLabel.FontColor = [1 1 1];
            app.MethodDropDownLabel.Position = [53 565 65 22];
            app.MethodDropDownLabel.Text = 'Method:';

            % Create StatusLamp
            app.StatusLamp = uilamp(app.RightPanel);
            app.StatusLamp.Position = [166 646 20 20];

            % Create StatusLabel
            app.StatusLabel = uilabel(app.RightPanel);
            app.StatusLabel.HorizontalAlignment = 'right';
            app.StatusLabel.FontColor = [1 1 1];
            app.StatusLabel.Position = [108 646 43 22];
            app.StatusLabel.Text = 'Status:';

            % Create NFFTEditField
            app.NFFTEditField = uieditfield(app.RightPanel, 'numeric');
            app.NFFTEditField.ValueChangedFcn = createCallbackFcn(app, @NFFTEditFieldValueChanged, true);
            app.NFFTEditField.Position = [132 468 100 22];

            % Create NFFTLabel
            app.NFFTLabel = uilabel(app.RightPanel);
            app.NFFTLabel.HorizontalAlignment = 'right';
            app.NFFTLabel.FontWeight = 'bold';
            app.NFFTLabel.FontColor = [1 1 1];
            app.NFFTLabel.Position = [78 468 39 22];
            app.NFFTLabel.Text = 'NFFT:';

            % Create ThresholdSEditField
            app.ThresholdSEditField = uieditfield(app.RightPanel, 'numeric');
            app.ThresholdSEditField.ValueChangedFcn = createCallbackFcn(app, @ThresholdSEditFieldValueChanged, true);
            app.ThresholdSEditField.Position = [132 440 100 22];

            % Create ThresholdSEditFieldLabel
            app.ThresholdSEditFieldLabel = uilabel(app.RightPanel);
            app.ThresholdSEditFieldLabel.HorizontalAlignment = 'right';
            app.ThresholdSEditFieldLabel.FontWeight = 'bold';
            app.ThresholdSEditFieldLabel.FontColor = [1 1 1];
            app.ThresholdSEditFieldLabel.Position = [38 440 79 22];
            app.ThresholdSEditFieldLabel.Text = 'Threshold S:';

            % Create ThresholdTEditField
            app.ThresholdTEditField = uieditfield(app.RightPanel, 'numeric');
            app.ThresholdTEditField.ValueChangedFcn = createCallbackFcn(app, @ThresholdTEditFieldValueChanged, true);
            app.ThresholdTEditField.Position = [132 412 100 22];

            % Create ThresholdTEditFieldLabel
            app.ThresholdTEditFieldLabel = uilabel(app.RightPanel);
            app.ThresholdTEditFieldLabel.HorizontalAlignment = 'right';
            app.ThresholdTEditFieldLabel.FontWeight = 'bold';
            app.ThresholdTEditFieldLabel.FontColor = [1 1 1];
            app.ThresholdTEditFieldLabel.Position = [40 412 77 22];
            app.ThresholdTEditFieldLabel.Text = 'Threshold T:';

            % Create UpdatePlotsButton
            app.UpdatePlotsButton = uibutton(app.RightPanel, 'push');
            app.UpdatePlotsButton.ButtonPushedFcn = createCallbackFcn(app, @UpdatePlotsButtonPushed, true);
            app.UpdatePlotsButton.BackgroundColor = [1 1 0];
            app.UpdatePlotsButton.FontWeight = 'bold';
            app.UpdatePlotsButton.Position = [241 412 84 78];
            app.UpdatePlotsButton.Text = {'Update'; 'Plots'};

            % Create SaveButton
            app.SaveButton = uibutton(app.RightPanel, 'push');
            app.SaveButton.ButtonPushedFcn = createCallbackFcn(app, @SaveButtonPushed3, true);
            app.SaveButton.Position = [241 36 97 24];
            app.SaveButton.Text = 'Save';

            % Create Image
            app.Image = uiimage(app.RightPanel);
            app.Image.Position = [223 704 83 56];
            app.Image.ImageSource = 'logo.png';

            % Create EditMenu
            app.EditMenu = uimenu(app.UIFigure);
            app.EditMenu.Text = 'Edit';

            % Create GlobalparametersMenu
            app.GlobalparametersMenu = uimenu(app.EditMenu);
            app.GlobalparametersMenu.MenuSelectedFcn = createCallbackFcn(app, @GlobalparametersMenuSelected, true);
            app.GlobalparametersMenu.Text = 'Global parameters';

            % Create FileMenu
            app.FileMenu = uimenu(app.UIFigure);
            app.FileMenu.Text = 'File';

            % Create OpenMenu
            app.OpenMenu = uimenu(app.FileMenu);
            app.OpenMenu.MenuSelectedFcn = createCallbackFcn(app, @OpenButtonPushed, true);
            app.OpenMenu.Text = 'Open';

            % Create SaveMenu
            app.SaveMenu = uimenu(app.FileMenu);
            app.SaveMenu.MenuSelectedFcn = createCallbackFcn(app, @SaveButtonPushed3, true);
            app.SaveMenu.Text = 'Save';

            % Create InfoMenu
            app.InfoMenu = uimenu(app.UIFigure);
            app.InfoMenu.Text = 'Info';

            % Create VersionLogMenu
            app.VersionLogMenu = uimenu(app.InfoMenu);
            app.VersionLogMenu.MenuSelectedFcn = createCallbackFcn(app, @VersionLogMenuSelected, true);
            app.VersionLogMenu.Text = 'Version Log';

            % Create AboutMenu
            app.AboutMenu = uimenu(app.InfoMenu);
            app.AboutMenu.MenuSelectedFcn = createCallbackFcn(app, @AboutMenuSelected, true);
            app.AboutMenu.Text = 'About...';

            % Create WebsiteMenu
            app.WebsiteMenu = uimenu(app.InfoMenu);
            app.WebsiteMenu.MenuSelectedFcn = createCallbackFcn(app, @WebsiteMenuSelected, true);
            app.WebsiteMenu.Text = 'Website';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = SiTraNo_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end