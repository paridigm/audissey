(

~knn = {|samp, training_data_file_path, dim|

	var data, smallest, data_smallest, dist;
	data = TabFileReader.read(training_data_file_path, true);
	smallest = nil;
	data_smallest = nil;



	//measure distance (get nearest neighbor for now)
	for(0, data.size-1,{|i|

		dist = ~get_dist.value(samp, data[i][1..dim], dim);

		//postln( (data[i]) ++ " " ++ dist.asFloat );

		if(smallest != nil) {
			if(dist < smallest) {
				smallest = dist;
				data_smallest = data[i];
			};
		}{
			smallest = ~get_dist.value(samp, data[i][1..dim], dim);
			data_smallest = data[i];
		};

		i = i+1;
	});

	data_smallest;

};


~knn2 = {|samp, training_data_file_path, dim, k, num_classes|

	var data, smallest, data_smallest, dist, max_class, placed, counter;
	data = TabFileReader.read(training_data_file_path, true);

	//smallest = Array.fill(k, { arg i; inf });
	smallest = Array.newClear(k);
	data_smallest = Array.newClear(k);
	max_class = Array.newClear(2); //[label, count]

	//measure distance (get nearest neighbor for now)
	for(0, data.size-1,{|i|

		dist = ~get_dist.value(samp, data[i][1..dim], dim);

		placed = false; //place inside k-smallest array
		for(0, k-1, {|j|

			if(placed == false) {

				if(smallest[j] != nil) {

					if(dist < smallest[j]) {

						/*
						if(j < k-1) {
							for(k-1, j+1, {|m| smallest[m]=smallest[m-1]; data_smallest[m]=data_smallest[m-1];}); //// shift ////
						};
						*/

						smallest[j] = dist;
						data_smallest[j] = data[i];
						placed = true;
					};

				}{
					smallest[j] = dist;
					data_smallest[j] = data[i];
					placed = true;
				};

			};

		});

		/*
		for(0, k-1,{|jj| postln(smallest[jj]);});
		for(0, k-1,{|jjj| if(data_smallest[jjj]!=nil) {post(data_smallest[jjj][0]);}; });
		postln("");
		*/

	});


	for(0, k-1,{|j| postln(data_smallest[j][0]);});


	//get most occuring class
	for(1, num_classes,{|i|

		counter=0; //counting class label occurances for class (i)
		for(0, k-1,{|j|

			if( data_smallest[j][0].asInt == (i).asInt ) {
				counter=counter+1;
			};

		});

		if(max_class[0] != nil) {
			if(counter > max_class[1]) {
				max_class[0] = i;
				max_class[1] = counter;
			};
		}{
			max_class[0] = i;
			max_class[1] = counter;
		};

		post(max_class[0].asString ++ "," ++ i.asString ++ " ");
		postln("");

	});//end counting

	max_class[0];

};


~get_dist = {|p1, p2, dim|

	var sum, temp;
	sum=0.0;
	for(0, dim-1,{|i|
		temp = abs(p2[i].asFloat - p1[i].asFloat);
		//postln("sqaure_diff: " ++ (temp*temp2).asFloat);
		sum = sum + (temp*temp);
		//postln("sum: " ++ sum.asString);
		i = i+1;
	});

	temp = sqrt(sum);
	temp;
};

)
