Identifying a person in crowd is very difficult taking conditions of real world, assuming the person is identified using survailence cameras we brought this solution,
making a two hand judgemental system which combines and analyses the probabilities and find the person in the crowd
This project uses face embedding and shirt colour mapping using extracting the color from region of intrest.
so the code accepts two inputs one the shirt color discription and other one is a reference image to the person that we want to identify

example input :
"A boy in blue shirt" or "blue shirt"
reference_image.png

The output is logged when the person is identified with time stamps

if only dress color got matched a green bounding box is drawn,if only face embedding matched a orange bounding box
A red bounding box is drawn only if both are face and dress are matched 


