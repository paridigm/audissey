(

~countdown_record_to_buffer = {|record_buffer, tempo=120, record_seconds=10|

	var rec_synth;
	var cond = Condition.new();

	//metronome sound
	SynthDef("metro", {|freq = 440| Out.ar( 0, SinOsc.ar(freq, 0, 0.5)*Line.kr(1,0,0.1,doneAction:2) ); }).add;

	//recording method
	rec_synth = SynthDef("tutorial-RecordBuf",{ arg out=0,bufnum=0;
		var noise;
		noise = SoundIn.ar(0);
		RecordBuf.ar(noise, bufnum, loop: 0, doneAction: 2);
	}).add;

	{
		r = Routine({

			1.do({
				x = Synth("metro", [freq: 70.midicps]);
				(60/tempo).wait;
				3.do({
					x = Synth("metro", [freq: 65.midicps]);
					(60/tempo).wait;
				});
			});

			x = Synth("metro", [freq: 70.midicps]);
			(60/tempo).wait;
			(3-2).do({
				x = Synth("metro", [freq: 65.midicps]);
				(60/tempo).wait;
			});

			cond.unhang;

		});

		r.play;
		cond.hang;

		postln("RECORDING...");
		rec_synth.play(s,[\out, 0, \bufnum, record_buffer]);


	}.fork;

	rec_synth; //return synth to stop it later

};

~create_score = {|onsets, commands, sound_names, params|

	var scr = Array.newClear(onsets.size+1); //score array

	for(0, onsets.size-1, {|j|

		if(params != nil) {
			scr[j] = [onsets[j], ["s_new", sound_names[commands[j]-1], (1000 + j), 0, 0, "freq", 100*j + 400]];

			//do this later
			/*
			for(0, params.size-1, {|i|
			scr[j][1][4*i + 3] = 0; scr[j][1][4*i + 3 + 1] = 0;
			scr[j][1][4*i + 3 + 2] = params[i][0];
			scr[j][1][4*i + 3 + 2] = params[i][1];
			}); */
			scr[j][1][3] = 0; scr[j][1][3 + 1] = 0;
			scr[j][1][3 + 2] = params[j][0].asString; postln(params[j][0]);
			scr[j][1][3 + 2] = params[j][1].asString; postln(params[j][1]);

		}{
			scr[j] = [onsets[j], ["s_new", sound_names[commands[j]-1], (1000 + j)]];
		};

		j = j + 1;
	});
	scr[onsets.size] = [0.0, [\c_set, 0, 0]];

	postln(scr);
	Score.new(scr);

};

//get timings according to onsets



)