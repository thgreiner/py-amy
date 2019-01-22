output "address" {
  value = "${aws_elb.web.dns_name}"
}

output "gpu_public_ip" {
  value = "${aws_spot_instance_request.web.public_ip}"
}

output "micro_public_ip" {
  value = "${aws_instance.micro.public_ip}"
}