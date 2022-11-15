# Face Recognition

Put picture with one person's face in `knowns` directory. Change the file name as the person's name like: `john.jpg` or `jane.jpg`. Then run `python face_recog_video.py --source "path" ` or `python face_recog_webcam.py`

### Saving Videos and Center Points

- Saved at the folder `recog_video`
- both .mp4 and .txt file is saved
- name of the file is the same as input file

### `face_recog_video.py` flag arguments

- `--source` : **`required`**, path to saved videos
- `--verbose` : `default == False`, if `--verbose True`, print the center points in each frame
- `--target` : `default == None`, if `--target "obama"`, save the points only for "obama". otherwise, save all.

### Example

`python face_recog_video --source data/obama.mp4`
`python face_recog_video --source data/obama.mp4 --verbose True, --target obama`
