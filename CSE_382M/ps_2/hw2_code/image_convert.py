import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ks = [1,2,4,16]
cnt = 0
norms_F = [0,0,0,0]


for k in ks:
    # Open the image using Pillow
    img = Image.open('grumpy.jpg').convert('RGB')
    img_array = np.array(img)
    # Get dimensions: height, width, channels
    h, w, c = img_array.shape


    # Prepare an array to store the reconstructed channels
    img_reconstructed = np.zeros_like(img_array, dtype=np.float64)

    # Process each channel individually
    for channel in range(c):
        # Get the 2D array for this channel
        X = img_array[:, :, channel]
        
        # Perform SVD on the channel matrix
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Truncate to rank k
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        
        norms_F[cnt] += np.linalg.norm(S_k, 'fro')**2

        # Reconstruct the channel
        X_reconstructed = U_k @ S_k @ Vt_k
        
        # Store the result
        img_reconstructed[:, :, channel] = X_reconstructed

    # Clip and convert back to uint8
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype('uint8')
    cnt += 1
    # Convert the array back to an image
    img_final = Image.fromarray(img_reconstructed, 'RGB')
    img_final.save('grumpy_k{}.jpg'.format(k))

# Compute the Frobenius norm of the original image
true_norm_F = 0
for channel in range(c):
    X = img_array[:, :, channel]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)
    true_norm_F += np.linalg.norm(S, 'fro')**2

print('Percentage of Frobenius norm preserved:')
for i in range(4):
    print('k = {}: {:.2f}%'.format(ks[i], norms_F[i] / true_norm_F * 100))

# Display the original image
# Load some example images (replace with your own images or arrays)
img1 = np.array(Image.open('grumpy_k1.jpg'))
img2 = np.array(Image.open('grumpy_k2.jpg'))
img3 = np.array(Image.open('grumpy_k4.jpg'))
img4 = np.array(Image.open('grumpy_k16.jpg'))

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Display each image and set a title
axes[0].imshow(img1)
axes[0].set_title('$k=1$')
axes[0].axis('off')  # Optionally turn off axis ticks
axes[0].text(0.5, -0.3, '$||A_k||_F/||A||_F = {:.2f}$'.format(norms_F[0]/true_norm_F), transform=axes[0].transAxes,
             ha='center', fontsize=12)

axes[1].imshow(img2)
axes[1].set_title('$k=2$')
axes[1].axis('off')
axes[1].text(0.5, -0.3, '$||A_k||_F/||A||_F = {:.2f}$'.format(norms_F[1]/true_norm_F), transform=axes[1].transAxes,ha='center', fontsize=12)

axes[2].imshow(img3)
axes[2].set_title('$k=4$')
axes[2].axis('off')
axes[2].text(0.5, -0.3, '$||A_k||_F/||A||_F = {:.2f}$'.format(norms_F[2]/true_norm_F), transform=axes[2].transAxes,ha='center', fontsize=12)

axes[3].imshow(img4)
axes[3].set_title('$k=16$')
axes[3].axis('off')
axes[3].text(0.5, -0.3, '$||A_k||_F/||A||_F = {:.2f}$'.format(norms_F[3]/true_norm_F), transform=axes[3].transAxes,ha='center', fontsize=12)

# Adjust layout and show the figure
plt.tight_layout()

plt.savefig('grumpy_final.jpg')
