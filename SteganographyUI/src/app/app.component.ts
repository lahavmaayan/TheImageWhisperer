// app.component.ts

import { Component, OnInit } from '@angular/core';
import {  FileUploader, FileSelectDirective } from 'ng2-file-upload/ng2-file-upload';

const URL = 'http://localhost:3000/api/upload';
const ENCRYPT = 1

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  public should_present_result_div = false;
  public result_title_style_class = "";
  public result = undefined;
  public icon_class = "";
  public result_text = "";

  public uploader: FileUploader = new FileUploader({url: URL, itemAlias: 'photo'});

  ngOnInit() {
    this.uploader.onAfterAddingFile = (file) => { 
      file.withCredentials = false;
      this.should_present_result_div = false;
     };
    this.uploader.onCompleteItem = (item: any, response: any, status: any, headers: any) => {
         console.log('ImageUpload:uploaded:', item, status, response);
         console.log(response);
         this.result = JSON.parse(response)["is_encrypt"];
         this.update_class_style();
         this.should_present_result_div = true;
        //  alert(JSON.parse(response)["result_text"]);
     };
 }
 update_class_style(){
   if (this.result == ENCRYPT){
    this.result_title_style_class = "text-warning";
    this.result_text = "Danger! 😵";
   } else{
     this.result_title_style_class = "text-success";
     this.result_text = "Nothing to worry about 🤓";
   };
 }
}
