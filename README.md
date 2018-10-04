# TheImageWhisperer
Data hack 2018

Required installation:
- steghide - you can install using this shitty official guide here:
             http://steghide.sourceforge.net/download.php

             If you're working on ubuntu - lucky you! You can install
             using this guide:
             https://scottlinux.com/2014/08/12/steganography-in-linux-from-the-command-line/

             If you're working on Mac - no luck for you, come back one year.

- Numpy
- opencv-python

In order to create your data sets:
- Download Cifar10 for python
- Create folders - big batch (in current location) -> stegged_folder, not_stegged_folder
- Import create_data_sets, run create_split_batches.

To run the frontend and backend (of the web site) you need:
1. angular6.
2. nodejs
3. run `npm install`
4. to run the node server, run - `nodemon server`
5. to run the angular (front-end), run - `ng serve`
6. visit `http://localhost:4200/` and start using The Image Whisperer! :)
