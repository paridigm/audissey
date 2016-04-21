/*
*
* loads wav back into the buffer and find the onsets
*
* once I have found the onsets, segment --> save into new wav files!
*
* get wave files --> make some sort of file containing training data w/ feature space
*
*/

//TODO: make functions:
/*
- make the training session save data + record settings in a txt file (done)
*/

s.boot;
~open_GUI.value();      //run init first
~peak_threshold = 0.01; //adjust to room

/////////////// init /////////////////////
(
{
var buf_fx=Array.newClear(3);//actual sounds
var cond = Condition.new();

//includes
thisProcess.interpreter.executeFile(thisProcess.nowExecutingPath.dirname +/+ "buffer_and_array_functions.scd");
thisProcess.interpreter.executeFile(thisProcess.nowExecutingPath.dirname +/+ "music_functions.scd");
thisProcess.interpreter.executeFile(thisProcess.nowExecutingPath.dirname +/+ "nrt_functions.scd");
thisProcess.interpreter.executeFile(thisProcess.nowExecutingPath.dirname +/+ "pr_functions.scd");

~record_seconds = 5;
~global_tempo = 120;
~global_tantum = 2;
~sb_const = s.sampleRate*60/(~global_tempo*~global_tantum); postln(~sb_const);

~session_type; //play=0, training=other
~global_local_path = thisProcess.nowExecutingPath.dirname;

c; //array for original audio
p; //peaks list
a; //class assignment list
~r; //starts list (was getting converted into a routine?)
n; //ends list
g; //list representing a section of a signal (TODO: make this into an array)

q; //peak array
t; //temp array for graphing peaks
u; //temp arrary for graphing peaks

b; //buf orig
d; //buf rectified
e; //buf rec-->smoothed
x; //buf for each individual sound! (hopefully this works)
h; //holds the recording synthdef while recording

~c1; //cepstrum coeff buffers
~c2;
~c3;
~c_MFCC;
~features;
f; //fft buffer (size = 1024)

l; //file object and file path
~player_name = "default";
~peak_threshold = 0.008;//default
~training_data_file_path = thisProcess.nowExecutingPath.dirname +/+ ~player_name ++ "_training_data.csv";

//server (inside a thread)
{

	~synth_names = ["ba", "hi", "sn"];

		buf_fx[0] =Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "bass_sample.wav", bufnum:16);
		SynthDef("ba", {|freq = 440| Out.ar( 0, PlayBuf.ar(1, buf_fx[0], doneAction:2) ); }).add; s.sync;
		buf_fx[1] =Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "hihat_sample.wav", bufnum:17);
		SynthDef("hi", {|freq = 640| Out.ar( 0, PlayBuf.ar(1, buf_fx[1], doneAction:2) ); }).add; s.sync;
		buf_fx[2] =Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "snare_sample.wav", bufnum:18);
		SynthDef("sn", {|freq = 740| Out.ar( 0, PlayBuf.ar(1, buf_fx[2], doneAction:2) ); }).add; s.sync;

	~session_type = 0;
	postln("session = " ++ ~session_type);

	//b = Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "beat_boxing_sample.wav");
	//b = Buffer.alloc(s, s.sampleRate*~record_seconds, 1, nil, 100);
	s.sync;
	postln("b allocated");

	c = Array.newClear(s.sampleRate*~record_seconds);
	postln("c allocated");

	d = Buffer.alloc(s, s.sampleRate*~record_seconds, 1, nil, 95);
	s.sync;
	postln("d allocated");

	e = Buffer.alloc(s, s.sampleRate*~record_seconds, 1, nil, 90);
	s.sync;
	postln("e allocated");

	f = Buffer.alloc(s, 1024, 1, nil, 85);
	s.sync;
	postln("f allocated");

	~c_MFCC = Array.newClear(13); //13 MFCC coeffs
	postln("c_MFCC allocated");

	~features = Array.newClear(13); //length = feature dimension
	postln("features allocated");

	a;
	p = List[];
	~r = List[];
	n = List[];
	g = Array.newClear(1);
	postln("a, p, r, and n, g allocated");

	q = Array.newClear(s.sampleRate*~record_seconds);
	postln("q allocated");

	t = Array.newClear(s.sampleRate*~record_seconds);
	u = Array.newClear(s.sampleRate*~record_seconds);

	cond.unhang;

}.fork;
cond.hang;

~process_recording = {
{

	var cond;
	var i_p; //preprocessing
	cond = Condition.new;

	//clear lists

	//open training data
	l = File(~training_data_file_path, "a");

	~buffer_to_array.value(b, c);
	s.sync;

	~rectify_array.value(c);
	~array_to_buffer.value(c, d);
	s.sync;

	~do_buffer_nrt.value(d, {|sig, rbuf, freq| RecordBuf.ar(LPF.ar(sig, freq), rbuf);}, c, nil, cond, false, 30);
	cond.hang;

	~find_max_peaks.value(c, p, ~peak_threshold); //p~peaks list (adjust peak threshold as necessary)

	//preprocess peaks to remove close peaks
		i_p = 0;
		while ( { i_p < (p.size-1) }, {
			if(p[i_p+1] - p[i_p] < 5000) {
				p.removeAt(i_p+1);
			}{
				i_p = i_p+1;
			};
		});

	~find_peak_slopes.value(c, p, 0.0025, ~r, n);

	//more preproccessing to remove small spikes (MFCC had nothing in them)
		i_p = 0;
		while ( { i_p < (p.size-1) }, {
			if(p[i_p] - ~r[i_p] < 200) { //should be ~100
				p.removeAt(i_p);
				~r.removeAt(i_p);
				n.removeAt(i_p);
			}{
				i_p = i_p+1;
			};
		});

	a=Array.newClear(p.size); //classification list
	s.sync;

	~max_peaks_to_array.value(p, q, cond);   //program could not handle all 3 of these
	//~max_peaks_to_array.value(r, t, cond);
	//~max_peaks_to_array.value(n, u, cond);
	s.sync;

	postln("moved peaks to arrays");

	//analyze each segment (onsets only)
	for(0, p.size-1, {|i|

		postln("analyzing segment");
		/*
		g = List[];
		~buffer_to_list_from_to.value(b, g, ~r[i], p[i]);
		s.sync;
		*/
		g = Array.newClear(p[i] - ~r[i] + 1);
		postln("SIZE: " ++ g.size.asString);

		~buffer_to_array_from_to.value(b, g, ~r[i], p[i]);    s.sync;
		x = Buffer.alloc(s, g.size, 1, nil, 77);              s.sync;
		~array_to_buffer.value(g, x);                         s.sync;

		//////////// extract features //////////////////
		for(0, 12, {|j| ~c_MFCC[j] = Buffer.alloc(s, floor(x.numFrames / ((1024/2/8))), 1, nil, 30+j); s.sync; j = j + 1;});
		~get_features_to_buffers_from.value(x, f, ~c_MFCC); (x.duration + 0.1).wait; //get more features?
		s.sync;

		postln("starting_feature_extraction");

		for(0, 12, {|j|
			~c_MFCC_a = FloatArray.newClear(  floor(x.numFrames/((1024/2/8))) );
			~buffer_to_array.value(~c_MFCC[j], ~c_MFCC_a);
			~features[j] = ~get_non_zero_ave.value(~c_MFCC_a);
			s.sync;
			j = j + 1;
		});

		postln("feature: " ++ ~features);

		///////////////// CLASSIFY OR TRAIN ////////////
		if(~session_type != 0) {

			a[i] = ~session_type;
			l.write(~session_type.asString  ++ "\t");
				for(0, 12, {|j|
					var temp = (~features[j]).asString;
					l.write(temp);
					l.write("\t");
					j = j + 1;});
		}{
			postln("CLASSIFYING");

			//features conatains the features
			a[i] = ~knn.value(~features, ~training_data_file_path, 3)[0];
			//a[i] = ~knn2.value(~features, ~training_data_file_path, 3, 3, 3);

		};
		///////////////////////////////////////////////

		if(~session_type != 0) {
			l.write("\n");
		};

		postln("seg" ++ i.asString ++ " complete");
		s.sync;

		i = i+1;
   });

	if(~session_type != 0) {
		l.write("\n\n"); //skip lines next training session
	};
	l.close;

	"DONE".postln;

	//snap nearest tick
	for(0, p.size-1, {|j|
				p[j] = ((p[j]/~sb_const).round*~sb_const);
		j = j + 1;
	});

	//playback!
	postln(p);
	postln(a);
	( ~create_score.value((p.asFloat)/(s.sampleRate), a.asInt, ~synth_names) ).play;

}.fork;
}; //end main function

////////////////////// GUI //////////////////////////////////
~open_GUI = {

var width = 400;
var height = 400;
var rec_button;
var play_button;
var name_text;
var training_buttons = [["play session",nil],["bass drum",nil],["hi hat",nil],["snare",nil]];

w = Window.new("Audissey", Rect(width, height, 500, 300)).background_(Color.green(1.3)).front;
//w.view.decorator = FlowLayout(w.view.bounds);

play_button =	Button.new(w, Rect(width - 10,200,100,20)).states_([[ "PLAY", Color.black, Color.green ]]).action_({
	b.play;
});


Button.new(w, Rect(width - 10,230,100,20)).states_([[ "BEAT", Color.black, Color.green(0.7) ]]).action_({
		~create_score.value((p.asFloat)/(s.sampleRate), a.asInt, ~synth_names).play;
		postln(~session_type);

});

Button.new(w, Rect(width - 10,260,100,20)).states_([[ "PLOT", Color.black, Color.green(0.7) ]]).action_({
		c.plot();
		q.plot();
});

name_text = TextField(w, Rect(20, 75, 150, 30));
name_text.action = {
			arg txt;
			~player_name = txt.value;
			~training_data_file_path = ~global_local_path +/+ ~player_name.asString ++ "_training_data.csv";
			txt.value.postln;
};
name_text.string = ~player_name;

for(0, training_buttons.size-1, {|i|
		training_buttons[i][1] =Button.new(w, Rect(45 ,3.8 + i*30,100,25)).states_([
			[ training_buttons[i][0], Color.black, Color.grey(1, 0.5) ],
			[ training_buttons[i][0], Color.black, Color.grey(0.7) ]]).action_({ arg obj;

			for(0, training_buttons.size-1, {|i| training_buttons[i][1].value = 0; });
			obj.value = 1;
			~session_type = i;
			postln("session type: " ++ (~session_type).asString);

		});
});
~session_type = 0;
training_buttons[0][1].value = 1;

Button.new(w, Rect(width/2 + 10,260,100,25)).states_([[ "PROCESS", Color.black, Color.gray ]]).action_({
		~process_recording.value();
});

rec_button = Button.new(w,Rect(220,130,80,80)).states_([["RECORD", Color.white, Color.red],
		                                                  ["STOP", Color.black, Color.white]]).action_({

	arg obj; //gets the button object
	obj.value.postln;

	if(obj.value == 1) {
		b = Buffer.alloc(s, s.sampleRate*~record_seconds, 1, nil, 100);
		h = ~countdown_record_to_buffer.value(b, ~global_tempo, ~record_seconds);
	}{
		//stop recording
		h.free;
		postln("DONE RECORDING");
		//b.play;
		//~process_recording.value();
	};

});

};

}.fork;


)

~open_GUI.value();


(
~set_new_training_data = {|name|



};

)


/////////////// ANALYSIS ///////////////
~c1.plot("cepstral coeff 1");
~c2.plot("cepstral coeff 2");
~c3.plot("cepstral coeff 3");

~c1_a; ~c2_a; ~c3_a;
(

~c1_a = Array.newClear(~c1.numFrames);
~c2_a = Array.newClear(~c1.numFrames);
~c3_a = Array.newClear(~c1.numFrames);

~buffer_to_array.value(~c1, ~c1_a);
~buffer_to_array.value(~c2, ~c2_a);
~buffer_to_array.value(~c3, ~c3_a);

)
(postln(""); postln(~c1_a); postln(~c2_a); postln(~c3_a); "";)
[~c1_a, ~c2_a, ~c3_a].flop.flat.plot("SNARE 2, MFCC's 1,2,3 (desc)" ,numChannels: 3);

~c1_a.size;
~c3_a.size;

x.play;

~buffer_to_list_from_to.value(b, g, r[2], n[2]);
~array_to_buffer.value(g, x);
x.play;

b.plot("signal");
q.plot("peaks");
t.plot("starts");
u.plot("ends");

b.play;
b.plot;

c.plot;
e.plot;
b.plot;
d.plot;

p.size;
r.size;
n.size;

//test
c[10000];
e.get(10000, {|msg| msg.postln});


////////////////// printing /////////////////////

//~print_array.value(c);
b.plot("b (buffer)");
c.plot("c (array)");
d.plot("d (buffer)");

//playback!
postln(p);
postln(a);
( ~create_score.value((p.asFloat)/(s.sampleRate), a.asInt, ~synth_names) ).play;
b.play;

