(

~buffer_to_array = {|buffer, array, anchor_cond|
	buffer.getToFloatArray(0,  wait:0.04, action:{ |array_passed|

		//load the array
      for(0, array_passed.size-1, {|i|
	      array[i] = array_passed[i];
	      i = i+1;
      });

		//postln("ARRAY_FILLED");

		if(anchor_cond != nil) {
			anchor_cond.unhang;
		};

	});//end action
};

~buffer_to_array_from_to = {|buffer, array, start_index, end_index|
	buffer.getToFloatArray(0,  wait:0.04, action:{ |array_passed|

		//load the list
      for(start_index, end_index, {|i|
			array[i - start_index] = array_passed[i];
			i = i+1;
      });

		//postln("ARRAY_FILLED");

	});//end action
};


~buffer_to_list_from_to = {|buffer, list, start_index, end_index|
	buffer.getToFloatArray(0,  wait:0.04, action:{ |array_passed|

		//ensure lists are empty
		list.clear();

		//load the list
      for(start_index, end_index, {|i|
			list.add(array_passed[i]);
	      i = i+1;
      });

		postln("LIST_FILLED");

	});//end action
};


~array_to_buffer = {|array, buffer|

buffer.loadCollection(array, action: {|buf|

		{PlayBuf.ar(buf.numChannels, buf.bufnum, BufRateScale.kr(buf.bufnum), loop: 0)};

		postln("BUFFER_FILLED");

	});//end action

};

~print_array = {|array|

	for(0, array.size-1, {|i|
		postln(array[i]);
		i = i+1;
	});

	postln("COMPLETE");

};


~rectify_array = {|array|

	for(0, array.size-1, {|i|
		array[i] = abs(array[i]);
		i = i+1;
	});

	postln("RECTIFIED");

};


~derive_array = {|array| //didn't work...
	var prev_2;
	var prev_1;
	prev_2 = array[0];
	for(1, array.size-2, {|i|
		prev_1 = array[i];
		array[i] = (prev_1 - prev_2)*10;
		prev_2 = prev_1;
		i = i+1;
	});
	postln("DERIVATIVE_COMPLETE");
};


~find_max_peaks = {|signal_array, peak_list, thresh|
	var max;
	var max_index;
	max = 0;
	max_index = 0;

	//ensure lists are empty
	peak_list.clear();

	for(0, signal_array.size-1, {|i|

		if(signal_array[i] > thresh) {
			//in a peak
			if(signal_array[i] > max) {
				max = signal_array[i];
				max_index = i;
			};
		}{
			//done with peak
			if(max > thresh) {
				peak_list.add(max_index);
				max = 0;
			};

		};

		i = i+1;
	});

	postln("PEAKS_FOUND");
};


~max_peaks_to_array = {|peak_list, array, anchor_cond|

	//set all to zero
	for(0, array.size-1, {|i| array[i] = 0; i = i+1;});

	//set peak locations
	for(0, peak_list.size-1, {|i|

		//NOTE: 200 so you can see it on a plot
		for(peak_list[i], peak_list[i]+200, {|j| array[j] = 1; j = i=j+1;});

		i = i+1;
	});

	postln("PEAKS_CONVERTED");
	anchor_cond.unhang;

};


~find_peak_slopes = {|env_array, peak_list, stop_thresh, starts_list, ends_list, smallest_limit|

	//ensure lists are empty
	starts_list.clear();
	ends_list.clear();

	if(smallest_limit == nil) {
		smallest_limit = 550;
	};

	//go through peak list and find the slopes going down to thresh for each peak given the envelope
	for(0, peak_list.size-1, {|i|

		//onset
		j = peak_list[i]; //index
		while ( { env_array[j] > stop_thresh || ((peak_list[i] - j) < (smallest_limit))}, {
			j = j - 1;
		});
		starts_list.add(j);

		//decay
		j = peak_list[i]; //index
		while ( { env_array[j] > stop_thresh }, {j = j + 1;});
		ends_list.add(j);

		i = i+1;
	});

	postln("STARTS_AND_ENDS_FOUND");
};


~get_non_zero_ave = {|signal_array|
	var ave;
	var non_zero_samples;
	ave = 0;
	non_zero_samples = 0;

	for(0, signal_array.size-1, {|i|

		if(signal_array[i] != 0 || signal_array[i] != nil) {

			ave = ave + signal_array[i];
			non_zero_samples = non_zero_samples + 1;

		};
		i = i+1;
	});

	ave = ave / non_zero_samples;

};

//deprecated
~smooth_buffer = {|in, out, cutoff|

	{

	SynthDef("LPF_filter_funct",{ arg out, input_buffer, output_buffer;

		   var signal;
		   signal = LPF.ar( PlayBuf.ar(1, input_buffer), cutoff );
		   RecordBuf.ar(signal, output_buffer, loop: 0);

		}//end action

	).play(s,[\out, 0, \input_buffer, in, \output_buffer, out]); //end synth def (I think "play" sends the recording to a bus)

	s.sync;

	postln("SMOOTHING_POSSIBLY_DONE"); //not true

	}.fork

};


~get_features_to_buffers_from = {|in, fft_buf_1024, cep_array|

	//assume that cep1, cep2, and cep3 are the correct length
	//fft_buf must be length 1024

	{

	SynthDef.new("cep_funct", {
			var signal, fft, array;

			//removes surrounding synthdef when done
			signal = PlayBuf.ar(1, in, BufRateScale.kr(in), doneAction: 2);

			//b is a small buffer for the fft window
			fft = FFT(fft_buf_1024, signal);
			array = MFCC.kr(fft); //--> spits out an array w/ mfcc coeffs based on the fft

			//Out.kr(0, array); //outputs the coeffs throughout time (channel = the coeffs (13 of the))
			Out.ar(0,Pan2.ar(signal)); //output sound to know where we are :)

			/*
			//get MFCC's from output proxy
			RecordBuf.kr(array[0], cep1, loop:0);
			RecordBuf.kr(array[1], cep2, loop:0);
			*/
			for(0, 12, {|j|
				RecordBuf.kr(array[j], cep_array[j], loop:0);
				j = j + 1;
			});

		}).play;

	s.sync;

	postln("CEPSTRUMS_POSSIBLY_IN_BUFFERS"); //not true

	}.fork

};


)
