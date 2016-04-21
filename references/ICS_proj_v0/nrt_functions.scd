(
//includes
thisProcess.interpreter.executeFile(thisProcess.nowExecutingPath.dirname +/+ "buffer_and_array_functions.scd");

//filtering abstractions (perhaps I can just make a function externally)
~perform_lpf = {|signal, result_buffer, param|
	var lpf;
	if(param == nil) { param = 440;};
	lpf = LPF.ar(signal, param);
	RecordBuf.ar(lpf, result_buffer);
};

~perform_hpf = {|signal, result_buffer, param|
	var hpf = HPF.ar(signal, 3000);
	RecordBuf.ar(hpf, result_buffer);
};

~perform_reverb = {|signal, result_buffer, param|
	var rev = FreeVerb.ar(signal);
	RecordBuf.ar(rev, result_buffer);
};

~perform_MFCC = {|signal, result_buffer, param|
	var fft = FFT(LocalBuf(1024, 1), signal);
	var array = MFCC.kr(fft);
	RecordBuf.kr(array[param], result_buffer, loop:0);
};


~do_buffer_nrt = {|buffer, perform_function, data, data_list, anchor_cond, playback, function_param|

	{
		var wav_str, sf, cond;
		cond = Condition.new;

		wav_str =  ~buffer_to_temp_wav.value(buffer); s.sync; //create temp wav file
		sf = SoundFile.openRead( wav_str ); sf.close;

		~do_nrt_on.value(wav_str, perform_function, data, data_list, cond, playback, function_param);
		cond.hang; //different condition

		File.delete(wav_str);

		if(anchor_cond != nil) {
			anchor_cond.unhang;
		}

	}.fork;

};


/**
  * Takes in a wav file and outputs an array. NOTE: anchor_cond is a Condition object.
  * add size param, add numchannels param (numChannels no)
  * for now I dont know how to use size when the list is useless (until I figure out how to dynamically set size)
  */
~do_nrt_on = {|wav_file_string, perform_function, data, data_list, anchor_cond, playback, function_param|

	var data_local, resultbuf, resultpath, oscpath, score, dur, sf, sf_out, cond, size, playback_buffer;
	cond = Condition.new;

	sf = SoundFile.openRead( wav_file_string );
	dur = sf.duration;
	sf.close;

	resultpath = PathName.tmp +/+ UniqueID.next ++ ".aiff"; //temp files
	oscpath = PathName.tmp +/+ UniqueID.next ++ ".osc";

	fork {

		score = Score([

			//TODO: set a size parameter (not sure how this is going to work for lists)
			[0, (resultbuf = Buffer.new(s, data.size, 1, 0)).allocMsg], //allocate buffer


			[0, [\d_recv, SynthDef(\onsets, {                                //OSC commands
				var sig = SoundIn.ar(0);

				//////////////////////////////

				perform_function.value(sig, resultbuf, function_param);

				/////////////////////////////

				//multichannel resultbuf to extract as many features as I want? -- nah

			}).asBytes]],

			[0, Synth.basicNew(\onsets, s, 1000).newMsg],
			[dur, resultbuf.writeMsg(resultpath, headerFormat: "AIFF", sampleFormat: "float")] //writeMsg to file

		]);

		// osc file path, output path, input path - input is soundfile to analyze
		score.recordNRT(oscFilePath: oscpath,   //outputs a binary data file
			outputFilePath: "NULL", //output soundfile about the osc binary
			inputFilePath:sf.path,  //get info about input file
			sampleRate: sf.sampleRate,
			options: ServerOptions.new
			.verbosity_(-1)
			.numInputBusChannels_(sf.numChannels)
			.numOutputBusChannels_(sf.numChannels)
			.sampleRate_(sf.sampleRate),
			action: { cond.unhang }  // this re-awakens the process afer NRT completes
		);
		cond.hang;  // wait for completion

		//now reopen the temp file...
		sf_out = SoundFile.openRead(resultpath);

		////////// get data /////////////
		data_local = FloatArray.newClear(sf_out.numFrames);
		sf_out.readData(data_local);

		if(data_list != nil) {
			//ensure empty and load
			data_list.clear();
			for(0, sf_out.numFrames, {|i|
				data_list.add(data_local[i]);
				i = i+1;
			});
		};

		if(data != nil) {
			//copy data_local into data
			for(0, data.size-1, {|i|
				data[i] = data_local[i];
				i = i+1;
			});
		};

		//I prefer a dynamic size output (for feature extraction)
		//maybe it's not needed...

		//////////////////////////////////

		sf_out.close;
		File.delete(oscpath); File.delete(resultpath);

		///////// test playback ///////////
		if(playback == true) {
			postln("TEST PLAYBACK:");
			postln("Data: " ++ data);
			playback_buffer = Buffer.alloc(s, sf_out.numFrames, 1); s.sync;
			~array_to_buffer.value(data, playback_buffer);          s.sync;
			playback_buffer.play;
			wait( sf_out.duration ); s.sync;
		};
		///////////////////////////////////

		postln("NRT COMPLETE");

		if(anchor_cond != nil) {
			anchor_cond.unhang;
		}

	};//end fork

};


/**
  * Be sure to sync!
  */
~buffer_to_temp_wav = {|buffer|

	var file_name = PathName.tmp +/+ UniqueID.next ++ ".wav";
	buffer.write(file_name, "WAVE", "int16");
	file_name;

};



)


