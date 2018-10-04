// server.js

const path = require('path');
const fs = require('fs');
const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser')
const app = express();
const router = express.Router();
var cmd = require('node-cmd');

// Use python shell
var pythonScriptPath = '../image_whisperer/main_flow.py';
var options = {
    args: [""]
};

const DIR = './uploads';
 
let storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, DIR);
    },
    filename: (req, file, cb) => {
      cb(null, file.originalname);
    }
});
let upload = multer({storage: storage});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
 
app.use(function (req, res, next) {
  res.setHeader('Access-Control-Allow-Origin', 'http://localhost:4200');
  res.setHeader('Access-Control-Allow-Methods', 'POST');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
  res.setHeader('Access-Control-Allow-Credentials', true);
  next();
});
 
app.get('/api', function (req, res) {
  res.end('file catcher example');
});
 
app.post('/api/upload',upload.single('photo'), function (req, res) {
    if (!req.file) {
        console.log("No file received");
        return res.send({
          success: false
        });
    
      } else {
        console.log('file received');
        filename = req.file.filename;
        command = "python " + pythonScriptPath + " " + filename;
        console.log(command);
        var result_text = ""
        cmd.get(command, function(err, data, stderr){
            if (err){
                console.error(err)
                console.log(stderr)
                return res.send({
                    success: false
                  })
            }
            console.log('this is the data : ',data)
            result_text = data
            return res.send({
                success: true,
                is_encrypt: data
      
              })
        });
        // PythonShell.run(pythonScriptPath, function (err, results) {
        //     if (err) throw err;
        //     // results is an array consisting of messages collected during execution
        //     console.log('results: %j', results);
        // });
      }
});
 
const PORT = process.env.PORT || 3000;
 
app.listen(PORT, function () {
  console.log('Node.js server is running on port ' + PORT);
});