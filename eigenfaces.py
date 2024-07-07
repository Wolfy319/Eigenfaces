import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Change from a 1xN^2 vector to a NxN matrix image
def vector_to_image(vector, dimension):
	return np.reshape(vector, (dimension,dimension))

# Access the face images and split them by person
def load_face_images(facefile, targetfile):
	# Load faces and face IDs
	faces = np.load(facefile)
	target = np.load(targetfile)
	curr_id = target[0]
	training_ims = []
	person = []
	for face, face_id in zip(faces, target):
		# New individual, start new list
		if face_id != curr_id:
			curr_id = face_id
			training_ims.append(person)
			person = [face.flatten()]
		# Same individual, add to their list
		else:
			person.append(face.flatten())
	# Add last set of faces
	training_ims.append(person)
	# Transform into matrix
	training_ims = np.asarray(training_ims)
	return training_ims, target, faces

# Create a face that represents the average of every training face
def find_average_face(training_ims, target):
	# Initialize empty matrix
	av_face = np.zeros_like(training_ims[0][0])
	# Access every face vector
	for person in training_ims:
		for face in person:
			# Add the face vector to a running sum vector
			av_face = np.add(av_face, face)
	# Divide the sum vector by the total number of faces
	av_face = av_face / len(target)
	return av_face

# Calculate each training face's variance from the average
def find_difference_faces(training_ims):
	difference_matrix = []
	# Access every face vector
	for person in training_ims:
		for face in person:
			# Calculate how much the face differs from the mean
			diff_im = np.subtract(face, av_face)
			# Add the difference vector to a matrix
			difference_matrix.append(diff_im)
			
	# Turn into numpy array, take transpose so row vectors become columns
	return np.asarray(difference_matrix).T

# Find vectors representing most prominent features of the training faces
def find_eigenfaces(difference_matrix, num_significant):

	# Calculate A(T)A instead of Covariance matrix AA(T)
	eigenmatrix = np.matmul(difference_matrix.T, difference_matrix)
	# Find the eigenvectors vi of A(T)A, sorted from heighest to lowest eigenvalue
	eigenvalues, eigenvectors = np.linalg.eig(eigenmatrix)

	# Calculate the eigenvectors of Covariance matrix AA(T), aka "eigenfaces"
  # Only keep the M' eigenfaces with the highest eigenvalue for efficiency
	eigenfaces = np.matmul(difference_matrix,eigenvectors)[:,:num_significant]
	# Normalize each eigenface so ||Avi|| = 1
	for i in range(num_significant):
		# Find vector norm
		norm = np.linalg.norm(eigenfaces[:,i])
		# Divide values in vector by its norm
		eigenfaces[:,i] = (1/norm) * eigenfaces[:,i] 
	return eigenfaces

# Find the contribution of each eigenface towards an image
def find_weights(face):
	# Create an empty vector, the pattern vector
	face_weights = []
	# Calculate difference of face from the mean
	diff = np.subtract(face, av_face)
	# Calculate the weight of each eigenface towards image, add to list
	for eface in eigenfaces.T:
		# Equivalent to doing the calculation eface(T)diff
		face_weights.append(np.matmul(eface, diff))
	# Turn into numpy array
	face_weights = np.asarray(face_weights)
	return face_weights

# Calculate the weights for every individual in the dataset
def classify_faces(authorized_faces):
	face_classes = []
	# Access every face vector
	for person in authorized_faces:
		weights = np.zeros_like(eigenfaces[0,:])
		for face in person:
			# Combine pattern vectors for several images of same person
			face_weights = find_weights(face)
			weights = np.add(weights, face_weights)
		# Take the average of the person's pattern vectors
		num_faces = len(person)
		weights = np.divide(weights, num_faces)
		# Add as a face classifier
		face_classes.append(weights)
	return face_classes

# Find the most similar face to an image
def distance_to_classes(image_pattern):
	distances = []
	# Look at every face class
	for classifier in face_classes:
		# Find how close image is to this class
		distance_to_class = np.linalg.norm(image_pattern - classifier)
		# Append nonzero distances
		if distance_to_class > 0:
			distances.append(distance_to_class)
	# Return index of and distance to closest face class
	closest_face = np.argmin(distances)
	min_distance = np.min(distances)
	return closest_face, min_distance

# Calculate a maximum distance threshold for 
# an image to be considered a match
def calculate_threshold():
	class_distances = []
	# Look at every face class
	for classifier in face_classes:
		# Find smallest distance from this class to all others
		_, dist = distance_to_classes(classifier)
		class_distances.append(dist)
	# Set threshold as 0.8 times the maximum of the minimum distances between classes
	return 0.8*np.max(class_distances)

# Decide if image is of authorized face, and return index of matching face
def is_authorized_face(image_vector):
	# Calculate eigenface weights
	image_pattern = find_weights(image_vector)
	# Find the face most similar to image and its distance
	closest_face, min_distance = distance_to_classes(image_pattern)
	# Return index of matching face if it's similar enough, otherwise return -1
	if min_distance < threshold:
		return closest_face, min_distance
	return -1, min_distance

# Reconstruct an image from eigenfaces
def recreate_image(pattern_vector):
  # Start with the average face
	image = av_face.copy()
	process = []
  # Access every weight and it's corresponding eigenface
	for weight, eface in zip(pattern_vector, eigenfaces.T):
    # Add the scaled eigenface to the average image
		process.append(image)
		image = np.add(image, weight * eface)
  # Return the recreated image and all the steps towards it
	return image, process

# Load face images
facefile = "./olivetti_faces.npy"
targetfile = "./olivetti_faces_target.npy"
authorized_faces, face_ids, face_images = load_face_images(facefile, targetfile)

# Split faces into authorized and unauthorized categories
authorized_faces = authorized_faces[:20]

# Perform PCA 
av_face = find_average_face(authorized_faces, face_ids)
difference_faces = find_difference_faces(authorized_faces)
eigenfaces = find_eigenfaces(difference_faces, num_significant=200)
# Classify the features of every authorized face
face_classes = classify_faces(authorized_faces)
# Decide the minimum allowable distance to a class for a face to be recognized
threshold = calculate_threshold()

decisions = []
distances = []
for face in face_images:
	face_vector = face.flatten()
  # Look at every face and determine if it's one from the training set
	decision = is_authorized_face(face_vector)
  # Add the decision made to a list
	decisions.append(decision[0])
	distances.append(decision[1])
 
accepted = []
rejected = []
# Sort distances into ones that were rejected and accepted
for i in range(len(distances)):
	if decisions[i] == -1:
		accepted.append(-1)
		rejected.append(distances[i])
	else:
		accepted.append(distances[i])
		rejected.append(-1)	

# Split decisions into the expected categories
auth = accepted[:199]
unauth = rejected[200:]

# Count how many authorized faces were seen as unauthorized
f_negative = auth.count(-1)

# Count how many unauthorized faces were seen as authorized
f_positive = unauth.count(-1)

# Calculate the recognition accuracy and print the values
accuracy = 100 - ((f_positive + f_negative) * 100 / 400)
print("False negatives: ", f_negative)
print("False positives: ", f_positive)
print("Accuracy: ", accuracy, "%")
